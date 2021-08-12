"""Module for the Database Creation."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import json
import logging
import pathlib
import time

import langdetect
import langdetect.lang_detect_exception
import pandas as pd
import sqlalchemy

from bluesearch.database.article import Article, CORD19ArticleParser
from bluesearch.utils import load_spacy_model

logger = logging.getLogger(__name__)


def mark_bad_sentences(engine, sentences_table_name):
    """Flag bad sentences in SQL database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        The connection to an SQL database.
    sentences_table_name : str
        The table with sentences.

    Raises
    ------
    RuntimeError
        If the column "is_bad" is missing in the table provided.
    """
    logger.info('Verifying the column "is_bad" is present')
    with engine.begin() as connection:
        inspector = sqlalchemy.inspect(connection)
        columns = inspector.get_columns(sentences_table_name)
    columns_name = [col["name"] for col in columns]
    if "is_bad" not in columns_name:
        raise RuntimeError("Column is_bad not found in given table")

    logger.info("Getting all sentences")
    with engine.begin() as connection:
        query = f"SELECT sentence_id, text FROM {sentences_table_name}"  # nosec
        df_sentences = pd.read_sql(query, connection)

    logger.info("Computing text lengths")
    text_length = df_sentences["text"].str.len()

    logger.info("Checking for LaTeX")
    has_latex = df_sentences["text"].str.contains(r"\\[a-z]+{")

    logger.info("Checking for minimal length")
    too_short = text_length < 20

    logger.info("Checking for maximal length")
    too_long = text_length > 2000

    df_sentences["is_bad"] = has_latex | too_short | too_long
    n_bad = df_sentences["is_bad"].sum()
    n_total = len(df_sentences)
    bad_percent = n_bad / n_total * 100
    logger.info(f"{n_bad} of {n_total} found to be bad ({bad_percent:.2f}%)")

    bad_sentence_ids = df_sentences["sentence_id"][df_sentences["is_bad"]]
    if len(bad_sentence_ids) > 0:
        logger.info("Writing results to database")
        bad_sentence_ids = ", ".join(str(id_) for id_ in bad_sentence_ids)
        with engine.begin() as connection:
            query = f"""
            UPDATE {sentences_table_name}
            SET is_bad = 1
            WHERE sentence_id in ({bad_sentence_ids})
            """
            connection.execute(query)
    else:
        logger.info("Nothing to write to database")


class CORD19DatabaseCreation:
    """Create SQL database from a specified dataset.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Directory to the dataset where metadata.csv and all jsons file
        are located.
    engine : SQLAlchemy.Engine
        Engine linked to the database.

    Attributes
    ----------
    max_text_length : int
        Max length of values in MySQL column of type TEXT. We have to
        constraint our text values to be smaller than this value
        (especially articles.abstract and sentences.text).
    """

    def __init__(self, data_path, engine):
        self.data_path = pathlib.Path(data_path)
        if not self.data_path.exists():
            raise NotADirectoryError(
                f"The data directory {self.data_path} does not exit"
            )

        self.metadata = pd.read_csv(self.data_path / "metadata.csv")
        self.is_constructed = False
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_text_length = 60000

    def construct(self):
        """Construct the database."""
        if not self.is_constructed:
            self._schema_creation()
            self.logger.info("Schemas of the tables are created.")
            self._articles_table()
            self.logger.info("Articles table is created.")
            self._sentences_table()
            self.logger.info("Sentences table is created.")
            self.is_constructed = True
        else:
            raise ValueError("This database is already constructed!")

    def _schema_creation(self):
        """Create the schemas of the different tables in the database."""
        metadata = sqlalchemy.MetaData()

        self.articles_table = sqlalchemy.Table(
            "articles",
            metadata,
            sqlalchemy.Column(
                "article_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
            ),
            sqlalchemy.Column("cord_uid", sqlalchemy.String(8), nullable=False),
            sqlalchemy.Column("sha", sqlalchemy.Text()),
            sqlalchemy.Column("source_x", sqlalchemy.Text()),
            sqlalchemy.Column("title", sqlalchemy.Text()),
            sqlalchemy.Column("doi", sqlalchemy.Text()),
            sqlalchemy.Column("pmcid", sqlalchemy.Text()),
            sqlalchemy.Column("pubmed_id", sqlalchemy.Text()),
            sqlalchemy.Column("license", sqlalchemy.Text()),
            sqlalchemy.Column("abstract", sqlalchemy.Text()),
            sqlalchemy.Column("publish_time", sqlalchemy.Date()),
            sqlalchemy.Column("authors", sqlalchemy.Text()),
            sqlalchemy.Column("journal", sqlalchemy.Text()),
            sqlalchemy.Column("mag_id", sqlalchemy.Text()),
            sqlalchemy.Column("who_covidence_id", sqlalchemy.Text()),
            sqlalchemy.Column("arxiv_id", sqlalchemy.Text()),
            sqlalchemy.Column("pdf_json_files", sqlalchemy.Text()),
            sqlalchemy.Column("pmc_json_files", sqlalchemy.Text()),
            sqlalchemy.Column("url", sqlalchemy.Text()),
            sqlalchemy.Column("s2_id", sqlalchemy.Text()),
            sqlalchemy.Column("is_english", sqlalchemy.Boolean()),
        )

        self.sentences_table = sqlalchemy.Table(
            "sentences",
            metadata,
            sqlalchemy.Column(
                "sentence_id",
                sqlalchemy.Integer(),
                primary_key=True,
                autoincrement=True,
            ),
            sqlalchemy.Column("section_name", sqlalchemy.Text()),
            sqlalchemy.Column(
                "article_id",
                sqlalchemy.Integer(),
                sqlalchemy.ForeignKey("articles.article_id"),
                nullable=False,
            ),
            sqlalchemy.Column("text", sqlalchemy.Text()),
            sqlalchemy.Column(
                "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
            ),
            sqlalchemy.Column(
                "sentence_pos_in_paragraph", sqlalchemy.Integer(), nullable=False
            ),
            sqlalchemy.UniqueConstraint(
                "article_id",
                "paragraph_pos_in_article",
                "sentence_pos_in_paragraph",
                name="sentence_unique_identifier",
            ),
            sqlalchemy.Column("is_bad", sqlalchemy.Boolean(), server_default="0"),
        )

        with self.engine.begin() as connection:
            metadata.create_all(connection)

    def _articles_table(self):
        """Fill the Article Table thanks to 'metadata.csv'.

        The articles table has all the metadata.csv columns
        expect the 'sha'. Moreover, the columns are renamed
        (cfr. _rename_columns).
        """
        rejected_articles = []
        df = self.metadata.drop_duplicates("cord_uid", keep="first")
        df["publish_time"] = pd.to_datetime(df["publish_time"])
        for index, article in df.iterrows():
            try:
                if (
                    isinstance(article["abstract"], str)
                    and len(article["abstract"]) > self.max_text_length
                ):
                    article["abstract"] = article["abstract"][: self.max_text_length]
                    self.logger.warning(
                        f"The abstract of article {index} has a length >"
                        f" {self.max_text_length} and was cut off for the "
                        f"database."
                    )
                with self.engine.begin() as con:
                    article.to_frame().transpose().to_sql(
                        name="articles", con=con, index=False, if_exists="append"
                    )
            except Exception as e:
                rejected_articles += [index]
                self.logger.error(
                    f"Number of articles rejected: {len(rejected_articles)}"
                )
                self.logger.error(f"Last rejected: {rejected_articles[-1]}")
                self.logger.error(str(e))

            if index % 1000 == 0:
                self.logger.info(f"Number of articles saved: {index}")

    def _process_article_sentences(self, article, nlp):
        paragraphs = []
        article_id = int(article["article_id"])
        paragraph_pos_in_article = 0
        pmc_json = pdf_json = False

        # Read title and abstract
        if article["title"] is not None:
            text = article["title"]
            meta = {
                "section_name": "Title",
                "article_id": article_id,
                "paragraph_pos_in_article": paragraph_pos_in_article,
            }
            paragraphs += [(text, meta)]
            paragraph_pos_in_article += 1
        if article["abstract"] is not None:
            text = article["abstract"]
            meta = {
                "section_name": "Abstract",
                "article_id": article_id,
                "paragraph_pos_in_article": paragraph_pos_in_article,
            }
            paragraphs += [(text, meta)]
            paragraph_pos_in_article += 1

        # Find files linked to articles
        if article["pmc_json_files"] is not None:
            pmc_json = True
            jsons_path = article["pmc_json_files"].split("; ")
        elif article["pdf_json_files"] is not None:
            pdf_json = True
            jsons_path = article["pdf_json_files"].split("; ")
        else:
            jsons_path = []

        # Load json
        for json_path in jsons_path:
            with open(self.data_path / json_path.strip()) as fp:
                json_file_data = json.load(fp)

            parser = CORD19ArticleParser(json_file_data)
            article = Article.parse(parser)
            for section_title, text in article.iter_paragraphs():
                metadata = {
                    "section_name": section_title,
                    "article_id": article_id,
                    "paragraph_pos_in_article": paragraph_pos_in_article,
                }
                paragraphs.append((text, metadata))
                paragraph_pos_in_article += 1

        sentences = self.segment(nlp, paragraphs)
        sentences_df = pd.DataFrame(
            sentences,
            columns=[
                "sentence_id",
                "section_name",
                "article_id",
                "text",
                "paragraph_pos_in_article",
                "sentence_pos_in_paragraph",
            ],
        )

        # Consider first n sentences in paper to quickly determine
        # if it is in English
        n_sents_language = 10
        is_english = self.check_is_english(
            " ".join(sentences_df[:n_sents_language]["text"])
        )
        update_stmt = """
        UPDATE articles
        SET is_english = :is_english
        WHERE article_id = :article_id
        """

        with self.engine.begin() as con:
            sentences_df.to_sql(
                name="sentences", con=con, index=False, if_exists="append"
            )
            con.execute(
                sqlalchemy.sql.text(update_stmt),
                is_english=is_english,
                article_id=article_id,
            )

        return pmc_json, pdf_json

    def _sentences_table(self, model_name="en_core_sci_lg"):
        """Fill the sentences table thanks to all the json files.

        For each paragraph, all sentences are extracted and populate
        the sentences table.

        Parameters
        ----------
        model_name : str, optional
            SpaCy model used to parse the text into sentences.

        Returns
        -------
        pmc :  int
            Number of articles with at least one pmc_json.
        pdf : int
            Number of articles that does not have pmc_json file but
            at least one pdf_json.
        rejected_articles : list of int
            Article_id of the articles that raises an error during
            the parsing.
        """
        nlp = load_spacy_model(model_name, disable=["tagger", "ner"])

        articles_table = pd.read_sql(
            """
            SELECT article_id, title, abstract, pmc_json_files, pdf_json_files
            FROM articles
            WHERE (abstract IS NOT NULL) OR (title IS NOT NULL)
            """,
            con=self.engine,
        )

        pdf = 0
        pmc = 0
        rejected_articles = []
        num_articles = 0
        start = time.perf_counter()

        for _, article in articles_table.iterrows():
            try:
                pmc_json, pdf_json = self._process_article_sentences(article, nlp)
                if pmc_json:
                    pmc += 1
                if pdf_json:
                    pdf += 1

            except Exception as e:
                rejected_articles += [int(article["article_id"])]
                self.logger.error(
                    f"{len(rejected_articles)} Rejected Articles: "
                    f"{rejected_articles[-1]}"
                )
                self.logger.error(str(e))

            num_articles += 1
            if num_articles % 1000 == 0:
                self.logger.info(
                    f"Number of articles: {num_articles} in "
                    f"{time.perf_counter() - start:.1f} seconds"
                )

        # Create article_id index
        mymodel_url_index = sqlalchemy.Index(
            "article_id_index", self.sentences_table.c.article_id
        )
        mymodel_url_index.create(bind=self.engine)

        # Create is_bad and article_id index
        sqlalchemy.Index(
            "is_bad_article_id_index",
            self.sentences_table.c.article_id,
            self.sentences_table.c.is_bad,
        ).create(bind=self.engine)

        # Create FULLTEXT INDEX
        if self.engine.url.drivername.startswith("mysql"):
            with self.engine.begin() as connection:
                self.logger.info(
                    "Start creating FULLTEXT INDEX on sentences (column text)"
                )
                connection.execute(
                    "CREATE FULLTEXT INDEX fulltext_text ON sentences(text)"
                )
                self.logger.info("Ended creating FULLTEXT INDEX")

        return pmc, pdf, rejected_articles

    def segment(self, nlp, paragraphs):
        """Segment a paragraph/article into sentences.

        Parameters
        ----------
        nlp : spacy.language.Language
            Spacy pipeline applying sentence segmentation.
        paragraphs : List of tuples (text, metadata)
            List of Paragraph/Article in raw text to segment into sentences.
            [(text, metadata), ].

        Returns
        -------
        all_sentences : list of dict
            List of all the sentences extracted from the paragraph.
        """
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]

        all_sentences = []
        for paragraph, metadata in nlp.pipe(paragraphs, as_tuples=True):
            for pos, sent in enumerate(paragraph.sents):
                text = str(sent)
                if len(text) > self.max_text_length:
                    text = text[: self.max_text_length]
                    self.logger.warning(
                        f'One sentence (article {metadata["article_id"]}, '
                        f'paragraph {metadata["paragraph_pos_in_article"]},'
                        f"sentence pos {pos}) has a length > {self.max_text_length}"
                        f"and was cut off for the database."
                    )
                all_sentences += [
                    {"text": text, "sentence_pos_in_paragraph": pos, **metadata}
                ]

        return all_sentences

    def check_is_english(self, text):
        """Check if the given text is English.

        Note the algorithm seems to be non-deterministic,
        as mentioned in https://github.com/Mimino666/langdetect#basic-usage.
        This is the reason of using `langdetect.DetectorFactory.seed = 0`

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        lang : bool or None
            Whether the language of the provided `text` is in English or not. If
            the input `text` is an empty string, `None` is returned.
        """
        langdetect.DetectorFactory.seed = 0
        lang = None
        if isinstance(text, str):
            try:
                lang = str(langdetect.detect(text))
            except langdetect.lang_detect_exception.LangDetectException as e:
                self.logger.info(e)

        is_english = lang == "en"
        return is_english

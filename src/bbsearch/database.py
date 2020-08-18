"""Module for the Database Creation."""
import json
import time

import pandas as pd
import spacy
import sqlalchemy


class CORD19DatabaseCreation:
    """Create SQL database from a specified dataset."""

    def __init__(self,
                 data_path,
                 engine):
        """Create SQL database object.

        Parameters
        ----------
        data_path: pathlib.Path
            Directory to the dataset where metadata.csv and all jsons file are located.
        engine: SQLAlchemy.Engine
            Engine linked to the database.
        """
        self.data_path = data_path
        if not self.data_path.exists():
            raise NotADirectoryError(f'The data directory {self.data_path} does not exit')

        self.metadata = pd.read_csv(self.data_path / 'metadata.csv')
        self.is_constructed = False
        self.engine = engine

    def construct(self):
        """Construct the database."""
        if not self.is_constructed:
            self._schema_creation()
            print('Schemas of the tables are created.')
            self._articles_table()
            print('Articles table is created.')
            self._sentences_table()
            print('Sentences table is created.')
            self.is_constructed = True
        else:
            raise ValueError('This database is already constructed!')

    def _schema_creation(self):
        """Create the schemas of the different tables in the database."""
        metadata = sqlalchemy.MetaData()

        self.articles_table = \
            sqlalchemy.Table('articles', metadata,
                             sqlalchemy.Column('article_id', sqlalchemy.Integer(),
                                               primary_key=True, autoincrement=True),
                             sqlalchemy.Column('cord_uid', sqlalchemy.String(8), nullable=False),
                             sqlalchemy.Column('sha', sqlalchemy.Text()),
                             sqlalchemy.Column('source_x', sqlalchemy.Text()),
                             sqlalchemy.Column('title', sqlalchemy.Text()),
                             sqlalchemy.Column('doi', sqlalchemy.Text()),
                             sqlalchemy.Column('pmcid', sqlalchemy.Text()),
                             sqlalchemy.Column('pubmed_id', sqlalchemy.Text()),
                             sqlalchemy.Column('license', sqlalchemy.Text()),
                             sqlalchemy.Column('abstract', sqlalchemy.Text()),
                             sqlalchemy.Column('publish_time', sqlalchemy.Date()),
                             sqlalchemy.Column('authors', sqlalchemy.Text()),
                             sqlalchemy.Column('journal', sqlalchemy.Text()),
                             sqlalchemy.Column('mag_id', sqlalchemy.Text()),
                             sqlalchemy.Column('who_covidence_id', sqlalchemy.Text()),
                             sqlalchemy.Column('arxiv_id', sqlalchemy.Text()),
                             sqlalchemy.Column('pdf_json_files', sqlalchemy.Text()),
                             sqlalchemy.Column('pmc_json_files', sqlalchemy.Text()),
                             sqlalchemy.Column('url', sqlalchemy.Text()),
                             sqlalchemy.Column('s2_id', sqlalchemy.Text())
                             )

        self.sentences_table = \
            sqlalchemy.Table('sentences', metadata,
                             sqlalchemy.Column('sentence_id', sqlalchemy.Integer(),
                                               primary_key=True, autoincrement=True),
                             sqlalchemy.Column('section_name', sqlalchemy.Text()),
                             sqlalchemy.Column('article_id', sqlalchemy.Integer(),
                                               sqlalchemy.ForeignKey("articles.article_id"),
                                               nullable=False),
                             sqlalchemy.Column('text', sqlalchemy.Text()),
                             sqlalchemy.Column('paragraph_pos_in_article', sqlalchemy.Integer(),
                                               nullable=False),
                             sqlalchemy.Column('sentence_pos_in_paragraph', sqlalchemy.Integer(),
                                               nullable=False),
                             sqlalchemy.UniqueConstraint('article_id',
                                                         'paragraph_pos_in_article',
                                                         'sentence_pos_in_paragraph',
                                                         name='sentence_unique_identifier')
                             )

        with self.engine.begin() as connection:
            metadata.create_all(connection)

        mymodel_url_index = sqlalchemy.Index('article_id_index', self.sentences_table.c.article_id)
        mymodel_url_index.create(bind=self.engine)

    def _articles_table(self):
        """Fill the Article Table thanks to 'metadata.csv'.

        The articles table has all the metadata.csv columns
        expect the 'sha'.
        Moreover, the columns are renamed (cfr. _rename_columns)
        """
        rejected_articles = []
        df = self.metadata.copy()
        df.drop_duplicates('cord_uid', keep='first', inplace=True)
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        for index, article in df.iterrows():
            try:
                article.to_frame().transpose().to_sql(name='articles', con=self.engine, index=False, if_exists='append')
            except Exception as e:
                print(e)
                rejected_articles += [index]
                print('Number of articles rejected: ', len(rejected_articles))
                print('Last rejected: ', rejected_articles[-1])

            if index % 1000 == 0:
                print('Number of articles saved: ', index)

    def _sentences_table(self, model_name='en_core_sci_lg'):
        """Fill the sentences table thanks to all the json files.

        For each paragraph, all sentences are extracted and populate
        the sentences table.

        Parameters
        ----------
        model_name: str, optional
            SpaCy model used to parse the text into sentences.

        Returns
        -------
        pmc: int
            Number of articles with at least one pmc_json.
        pdf: int
            Number of articles that does not have pmc_json file but at least one pdf_json.
        rejected_articles: list of int
            Article_id of the articles that raises an error during the parsing.
        """
        nlp = spacy.load(model_name, disable=["tagger", "ner"])

        articles_table = pd.read_sql("""SELECT article_id, title, abstract, pmc_json_files, pdf_json_files
                                        FROM articles
                                        WHERE (abstract IS NOT NULL) OR (title IS NOT NULL)""",
                                     con=self.engine)

        pdf = 0
        pmc = 0
        rejected_articles = []
        num_articles = 0
        start = time.perf_counter()

        for _, article in articles_table.iterrows():
            try:
                paragraphs = []
                article_id = int(article['article_id'])
                paragraph_pos_in_article = 0

                # Read title and abstract
                if article['title'] is not None:
                    paragraphs += [(article['title'], {'section_name': 'Title', 'article_id': article_id,
                                                       'paragraph_pos_in_article': paragraph_pos_in_article})]
                    paragraph_pos_in_article += 1
                if article['abstract'] is not None:
                    paragraphs += [(article['abstract'], {'section_name': 'Abstract', 'article_id': article_id,
                                                          'paragraph_pos_in_article': paragraph_pos_in_article})]
                    paragraph_pos_in_article += 1

                # Find files linked to articles
                if article['pmc_json_files'] is not None:
                    pmc += 1
                    jsons_path = article['pmc_json_files'].split('; ')
                elif article['pdf_json_files'] is not None:
                    pdf += 1
                    jsons_path = article['pdf_json_files'].split('; ')
                else:
                    jsons_path = []

                # Load json
                for json_path in jsons_path:
                    json_path = self.data_path / json_path.strip()
                    with open(str(json_path), 'r') as json_file:
                        file = json.load(json_file)

                        for paragraph_pos_in_article, section in enumerate(file['body_text'],
                                                                           start=paragraph_pos_in_article):
                            paragraphs += [(section['text'], {'section_name': section['section'].title(),
                                                              'article_id': article_id,
                                                              'paragraph_pos_in_article': paragraph_pos_in_article})]
                            paragraph_pos_in_article += 1

                        for paragraph_pos_in_article, (_, v) in enumerate(file['ref_entries'].items(),
                                                                          start=paragraph_pos_in_article):
                            paragraphs += [(v['text'], {'section_name': 'Caption', 'article_id': article_id,
                                                        'paragraph_pos_in_article': paragraph_pos_in_article})]

                            paragraph_pos_in_article += 1

                sentences = self.segment(nlp, paragraphs)
                sentences_df = pd.DataFrame(sentences, columns=['sentence_id', 'section_name', 'article_id',
                                                                'text', 'paragraph_pos_in_article',
                                                                'sentence_pos_in_paragraph'])
                sentences_df.to_sql(name='sentences', con=self.engine, index=False, if_exists='append')

            except Exception:
                rejected_articles += [int(article['article_id'])]
                print(len(rejected_articles), 'Rejected Articles:', rejected_articles[-1])

            num_articles += 1
            if num_articles % 1000 == 0:
                print('Number of articles: ', num_articles,
                      'in', f'{time.perf_counter() - start:.1f} seconds')

        return pmc, pdf, rejected_articles

    @staticmethod
    def segment(nlp, paragraphs):
        """Segment a paragraph/article into sentences.

        Parameters
        ----------
        nlp: spacy.language.Language
            Spacy pipeline applying sentence segmentation.
        paragraphs: List of tuples (text, metadata)
            List of Paragraph/Article in raw text to segment into sentences. [(text, metadata), ]

        Returns
        -------
        all_sentences: list of dict
            List of all the sentences extracted from the paragraph.
        """
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs, ]

        all_sentences = []
        for paragraph, metadata in nlp.pipe(paragraphs, as_tuples=True):
            for pos, sent in enumerate(paragraph.sents):
                all_sentences += [{'text': str(sent), 'sentence_pos_in_paragraph': pos, **metadata}]

        return all_sentences

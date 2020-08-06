"""SQL Related functions."""
import logging

import numpy as np
import pandas as pd


def retrieve_sentences_from_sentence_ids(sentence_ids, engine):
    """Retrieve sentences given sentence ids.

    Parameters
    ----------
    sentence_ids: list of int
        Sentence ids for which need to retrieve the text.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    sentences: pd.DataFrame
        Pandas DataFrame containing all sentences and their corresponding metadata:
        article_id, sentence_id, section_name, text, paragraph_pos_in_article.
    """
    sentence_ids_s = ', '.join(str(id_) for id_ in sentence_ids)
    sql_query = f"""SELECT article_id, sentence_id, section_name, text, paragraph_pos_in_article
                    FROM sentences
                    WHERE sentence_id IN ({sentence_ids_s})"""
    sentences = pd.read_sql(sql_query, engine)

    return sentences


def retrieve_paragraph_from_sentence_id(sentence_id, engine):
    """Retrieve paragraph given one sentence id.

    Parameters
    ----------
    sentence_id: int
        Sentence id for which need to retrieve the paragraph.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    paragraph: str or None
        If ``str`` then a paragraph containing the sentence of the given sentence_id. If None
        then the `sentence_id` was not found in the sentences table.
    """
    sql_query = f"""SELECT text
                    FROM sentences
                    WHERE article_id =
                        (SELECT article_id
                        FROM sentences
                        WHERE sentence_id = {sentence_id})
                    AND paragraph_pos_in_article =
                        (SELECT paragraph_pos_in_article
                        FROM sentences
                        WHERE sentence_id = {sentence_id})
                    ORDER BY sentence_pos_in_paragraph ASC"""

    all_sentences = pd.read_sql(sql_query, engine)['text'].to_list()
    if not all_sentences:
        paragraph = None
    else:
        paragraph = ' '.join(all_sentences)
    return paragraph


def retrieve_paragraph(article_id, paragraph_pos_in_article, engine):
    """Retrieve paragraph given one identifier (article_id, paragraph_pos_in_article).

    Parameters
    ----------
    article_id : int
        Article id.

    paragraph_pos_in_article : int
        Relative position of a paragraph in an article. Note that the numbering starts from 0.

    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    paragraph: pd.DataFrame
        pd.DataFrame with the paragraph and its metadata:
        article_id, text, section_name, paragraph_pos_in_article.
    """
    sql_query = f"""SELECT section_name, text
                    FROM sentences
                    WHERE article_id = {article_id}
                    AND paragraph_pos_in_article = {paragraph_pos_in_article}
                    ORDER BY sentence_pos_in_paragraph ASC"""

    sentences = pd.read_sql(sql_query, engine)
    if sentences.empty:
        paragraph = pd.DataFrame(columns=['article_id', 'text',
                                          'section_name', 'paragraph_pos_in_article'])
    else:
        sentences_text = sentences['text'].to_list()
        section_name = sentences['section_name'].iloc[0]
        paragraph_text = ' '.join(sentences_text)

        paragraph = pd.DataFrame([{'article_id': article_id,
                                   'text': paragraph_text,
                                   'section_name': section_name,
                                   'paragraph_pos_in_article': paragraph_pos_in_article}, ])
    return paragraph


def retrieve_article_metadata_from_article_id(article_id, engine):
    """Retrieve article metadata given one article id.

    Parameters
    ----------
    article_id: int
        Article id for which need to retrieve the article metadata.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article: pd.DataFrame
        DataFrame containing the article metadata. The columns are 'article_id', 'cord_uid', 'sha',
        'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
        'publish_time', 'authors', 'journal', 'mag_id', 'who_covidence_id', 'arxiv_id',
        'pdf_json_files', 'pmc_json_files', 'url', 's2_id'.
    """
    sql_query = f"""SELECT *
                    FROM articles
                    WHERE article_id = {article_id}"""
    article = pd.read_sql(sql_query, engine)
    return article


def retrieve_articles(article_ids, engine):
    """Retrieve article given multiple article ids.

    Parameters
    ----------
    article_ids: list of int
        List of Article id for which need to retrieve the entire text article.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    articles: pd.DataFrame
        DataFrame containing the articles divided into paragraphs. The columns are
        'article_id', 'paragraph_pos_in_article', 'text', 'section_name'.
    """
    articles_str = ', '.join(str(id_) for id_ in article_ids)
    sql_query = f"""SELECT *
                    FROM sentences
                    WHERE article_id IN ({articles_str})
                    ORDER BY article_id ASC,
                    paragraph_pos_in_article ASC,
                    sentence_pos_in_paragraph ASC"""
    all_sentences = pd.read_sql(sql_query, engine)

    groupby_var = all_sentences.groupby(by=['article_id', 'paragraph_pos_in_article'])
    paragraphs = groupby_var['text'].apply(lambda x: ' '.join(x))
    section_name = groupby_var['section_name'].unique().apply(lambda x: x[0])

    articles = pd.DataFrame({'text': paragraphs,
                             'section_name': section_name}).reset_index()

    return articles


class SentenceFilter:
    """Filter sentence IDs by applying conditions.

    Instantiate this class and apply different filters by calling
    the corresponding filtering methods in any order. Finally,
    call either the `run()` or the `stream()` method to obtain
    the filtered sentence IDs.

    Example
    -------

    .. code-block:: python

        import sqlalchemy
        connection = sqlalchemy.create_engine("...")
        filtered_sentence_ids = (
            SentenceFilter(connection)
            .only_with_journal()
            .restrict_sentences_ids_to([1, 2, 3, 4, 5])
            .date_range((2010, 2020))
            .exclude_strings(["virus", "disease"])
            .run()
        )

    When the `run()` or the `stream()` method is called an SQL
    query is constructed and executed internally. For the example
    above it would have approximately the following form

    .. code-block:: SQL

        SELECT sentence_id
        FROM sentences
        WHERE
            article_id IN (
                SELECT article_id
                FROM articles
                WHERE
                    publish_time BETWEEN '2010-01-01' AND '2020-12-31' AND
                    journal IS NOT NULL
            ) AND
            sentence_id IN ('1', '2', '3', '4', '5') AND
            text NOT LIKE '%virus%' AND
            text NOT LIKE '%disease%'

    Parameters
    ----------
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database that contains the `articles`
        and `sentences` tables.
    """

    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(self.__class__.__name__)

        self.only_with_journal_flag = False
        self.year_from = None
        self.year_to = None
        self.string_exclusions = []
        self.restricted_sentence_ids = None

    def only_with_journal(self, flag=True):
        """Only select articles with a journal.

        Parameters
        ----------
        flag : bool
            If True, then only articles for which a journal was
            specified will be selected.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Only with journal: {flag}")
        self.only_with_journal_flag = flag
        return self

    def date_range(self, date_range=None):
        """Restrict to articles in a given date range.

        Parameters
        ----------
        date_range : tuple or None
            A tuple with two elements of the form `(start_year, end_year)`.
            If None then nothing no date range is applied.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Date range: {date_range}")
        if date_range is not None:
            self.year_from, self.year_to = date_range
        return self

    def exclude_strings(self, strings):
        """Exclude sentences containing any of the given strings.

        Parameters
        ----------
        strings : list_like
            The strings to exclude.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        self.logger.info(f"Exclude strings: {strings}")
        strings = map(lambda s: s.lower(), strings)
        strings = filter(lambda s: len(s) > 0, strings)
        self.string_exclusions.extend(strings)
        return self

    def restrict_sentences_ids_to(self, sentence_ids):
        """Restrict sentence IDs to the given ones.

        Parameters
        ----------
        sentence_ids : list_like
            The sentence IDs to restrict to.

        Returns
        -------
        self : SentenceFilter
            The instance of `SentenceFilter` itself. Useful for
            chained applications of filters.
        """
        # For logging
        if len(sentence_ids) > 5:
            ids_str = f"[{', '.join(map(str, sentence_ids[:5]))} ..."
        else:
            ids_str = str(sentence_ids)
        self.logger.info(f"Restricting to sentencs IDs: {ids_str}")

        # The actual restriction
        self.restricted_sentence_ids = tuple(sentence_ids)

        return self

    def _build_query(self):
        article_conditions = []
        sentence_conditions = []

        # Journal condition
        if self.only_with_journal_flag:
            article_conditions.append("journal IS NOT NULL")

        # Date range condition
        if self.year_from is not None and self.year_to is not None:
            from_date = f"{self.year_from:04d}-01-01"
            to_date = f"{self.year_to:04d}-12-31"
            article_conditions.append(
                f"publish_time BETWEEN '{from_date}' AND '{to_date}'"
            )

        # Add article conditions to sentence conditions
        if len(article_conditions) > 0:
            article_condition_query = f"""
            article_id IN (
                SELECT article_id
                FROM articles
                WHERE {" AND ".join(article_conditions)}
            )
            """.strip()
            sentence_conditions.append(article_condition_query)

        # Restricted sentence IDs
        if self.restricted_sentence_ids is not None:
            sentence_ids_s = ", ".join(str(x) for x in self.restricted_sentence_ids)
            sentence_conditions.append(f"sentence_id IN ({sentence_ids_s})")

        # Exclusion text
        for text in self.string_exclusions:
            sentence_conditions.append(f"text NOT LIKE '%{text}%'")

        # Build and send query
        query = "SELECT sentence_id FROM sentences"
        if len(sentence_conditions) > 0:
            query = f"{query} WHERE {' AND '.join(sentence_conditions)}"

        return query

    def iterate(self, chunk_size):
        """Run the filtering query and iterate over restricted sentence IDs.

        Parameters
        ----------
        chunk_size : int
            The size of the batches of sentence IDs that are yielded.

        Yields
        ------
        result_arr : np.ndarray
            A 1-dimensional numpy array with the filtered sentence IDs.
            Its length will be at most equal to `chunk_size`.
        """
        self.logger.info(f"Iterating filtering with chunk size {chunk_size}")

        query = self._build_query()
        # self.logger.info(f"Query: {query}")
        for df_results in pd.read_sql(query, self.connection, chunksize=chunk_size):
            result_arr = df_results["sentence_id"].to_numpy()
            yield result_arr

    def run(self):
        """Run the filtering query to find restricted sentence IDs.

        Returns
        -------
        result_arr : np.ndarray
            A 1-dimensional numpy array with the filtered sentence IDs.
        """
        self.logger.info("Running the filtering query")

        query = self._build_query()
        # self.logger.info(f"Query: {query}")

        self.logger.debug("Running pd.read_sql")
        results = [row[0] for row in self.connection.execute(query).fetchall()]

        self.logger.info(f"Filtering gave {len(results)} results")

        return np.array(results)

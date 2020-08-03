"""SQL Related functions."""
import pandas as pd


def retrieve_sentences_from_sentence_id(sentence_id, engine):
    """Retrieve sentences given sentence ids.

    Parameters
    ----------
    sentence_id: list of int
        Sentences id for which need to retrieve the text.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    sentences: pd.DataFrame
        Pandas DataFrame containing all sentences and their corresponding metadata.
    """
    sentences_id = ', '.join(str(id_) for id_ in sentence_id)
    sql_query = f"""SELECT article_id, sentence_id, section_name, text, paragraph_pos_in_article
                    FROM sentences
                    WHERE sentence_id IN ({sentences_id})"""
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
    paragraph: str
        Paragraph containing the sentence of the given sentence_id.
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


def retrieve_paragraph(identifier, engine):
    """Retrieve paragraph given one identifier (article_id, paragraph_pos_in_article).

    Parameters
    ----------
    identifier: tuple of int
        Tuple with form: (Article_id, paragraph_pos_in_article)
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    paragraph: pd.DataFrame
        pd.DataFrame with the paragraph and the metadata.
    """
    sql_query = f"""SELECT section_name, text
                    FROM sentences
                    WHERE article_id = {identifier[0]}
                    AND paragraph_pos_in_article = {identifier[1]}
                    ORDER BY sentence_pos_in_paragraph ASC"""

    sentences = pd.read_sql(sql_query, engine)
    if sentences.empty:
        paragraph = pd.DataFrame(columns=['article_id', 'text',
                                          'section_name', 'paragraph_pos_in_article'])
    else:
        sentences_text = sentences['text'].to_list()
        section_name = sentences['section_name'].iloc[0]
        paragraph_text = ' '.join(sentences_text)

        paragraph = pd.DataFrame([{'article_id': identifier[0],
                                   'text': paragraph_text,
                                   'section_name': section_name,
                                   'paragraph_pos_in_article': identifier[0]}, ])
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
        DataFrame containing the article metadata from
        which the sentence is coming.
    """
    sql_query = f"""SELECT *
                    FROM articles
                    WHERE article_id = {article_id}"""
    article = pd.read_sql(sql_query, engine)
    return article


def retrieve_article(article_id, engine):
    """Retrieve article given one article id.

    Parameters
    ----------
    article_id: list of int
        List of Article id for which need to retrieve the entire text article.
    engine: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article: pd.DataFrame
        DataFrame containing the Article divided into paragraphs.
    """
    articles_str = ', '.join(str(id_) for id_ in article_id)
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


def get_ids_by_condition(conditions, table, db_cnxn):
    """Find entry IDs given a number of search conditions.

    Notes
    -----
    In the database 'cord19', tables are named with plural noun (e.g sentences, articles)
    However, column id are named in singular form (e.g. sentence, article)

    Parameters
    ----------
    conditions : list
        A list strings representing SQL query conditions. They should be
        formatted so that they can be used in an SQL WHERE statement,
        for example:
            SELECT * FROM {table} WHERE <condition_1> and <condition_2>"
    table : str
        The name of the table in `db`.
    db_cnxn : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        A SQL database for querying the article IDs. Should contain
        a table named "articles".

    Returns
    -------
    results : list
        A list of article IDs (=SHAs) represented as strings.
    """
    if conditions:
        condition = ' and '.join(conditions)
        sql_query = f"SELECT {table[:-1]}_id FROM {table} WHERE {condition}"
    else:
        sql_query = f"SELECT {table[:-1]}_id FROM {table}"

    results = pd.read_sql(sql_query, db_cnxn)[f'{table[:-1]}_id'].tolist()

    return results


def get_sentence_ids_by_condition(conditions, db_cnxn, sentence_ids=None):
    """Get ids of all sentences satisfying a set of conditions.

    Parameters
    ----------
    conditions : list
        A list strings representing SQL query conditions. They should be
        formatted so that they can be used in an SQL WHERE statement,
        for example:
            SELECT * FROM {table} WHERE <condition_1> and <condition_2>"

    db_cnxn : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        A SQL database for querying the article IDs. Should contain
        a table named "sentences"

    sentence_ids : list or None
        Initial preselection of sentence_ids to considered. It can speed up the search
        drastically. If None then not considered.

    Returns
    -------
    results : list
        A list of sentence ids.
    """
    if sentence_ids is not None:
        sentence_ids_s = ', '.join([str(x) for x in sentence_ids])
        subtable = f"""(SELECT * from sentences WHERE sentence_id IN ({sentence_ids_s})) AS t1"""
    else:
        subtable = "sentences"

    condition = ' and '.join(conditions)

    if condition:
        condition = ' and '.join(conditions)
        sql_query = f"SELECT sentence_id FROM {subtable} WHERE {condition}"
    else:
        sql_query = f"SELECT sentence_id FROM {subtable}"

    results = pd.read_sql(sql_query, db_cnxn)['sentence_id'].tolist()

    return results


class ArticleConditioner:
    """Article conditioner."""

    @staticmethod
    def get_date_range_condition(date_range):
        """Construct a date range condition.

        Notes
        -----
        Rows without a date entry will never be selected (to be checked)

        Parameters
        ----------
        date_range : tuple
            A tuple of the form (date_from, date_to)

        Returns
        -------
        condition : str
            The SQL condition
        """
        date_from, date_to = date_range
        condition = f"publish_time BETWEEN '{date_from}-01-01' and '{date_to}-12-31'"

        return condition

    @staticmethod
    def get_has_journal_condition():
        """Construct a has-journal condition.

        Returns
        -------
        condition : str
            The SQL condition
        """
        condition = "journal IS NOT NULL"

        return condition

    @staticmethod
    def get_restrict_to_tag_condition(tag):
        """Construct a condition for restricting to a given tag.

        Parameters
        ----------
        tag : str
            The tag to constrain on

        Returns
        -------
        condition : str
            The SQL condition
        """
        condition = f"{tag} = 1"
        return condition


class SentenceConditioner:
    """Sentence conditioner."""

    @staticmethod
    def get_article_id_condition(article_id):
        """Construct condition for specific article_id.

        Parameters
        ----------
        article_id: list

        Returns
        -------
        condition
        """
        articles = ', '.join(str(id_) for id_ in article_id)
        condition = f"article_id IN ({articles})"
        return condition

    @staticmethod
    def get_word_exclusion_condition(word):
        """Construct condition for exclusion containing a given word.

        Parameters
        ----------
        word

        Returns
        -------
        condition
        """
        condition = f"text NOT LIKE '%{word}%'"
        return condition

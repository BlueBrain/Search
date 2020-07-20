"""
SQL Related functions.
"""
import pandas as pd


def retrieve_sentences_from_sentence_id(sentence_id, engine):
    """Retrieve sentences given sentence ids.

    Parameters
    ----------
    sentence_id: list of int
        Sentences id for which need to retrieve the text.
    engine: sqlalchemy.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    texts: pd.DataFrame
        Pandas DataFrame containing sentence_id and corresponding text.
    """
    sentences_id = ', '.join(str(id_) for id_ in sentence_id)
    sql_query = f'SELECT sentence_id, text FROM sentences WHERE sentence_id IN ({sentences_id})'
    sentences = pd.read_sql(sql_query, engine)

    return sentences


def retrieve_sentences_from_section_name(section_name, engine):
    """Retrieve sentences given section names.

    Parameters
    ----------
    section_name: list of str
        Sentences id for which need to retrieve the text.
    engine: sqlalchemy.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    sentences: pd.DataFrame
        DataFrame containing the sentences and their sentence_id
        coming from the given section name.
    """
    section_names = ' ,'.join(f"'{name}'" for name in section_name)
    sql_query = f'SELECT sentence_id, section_name, text FROM sentences WHERE section_name IN ({section_names})'
    sentences = pd.read_sql(sql_query, engine)
    return sentences


def retrieve_article_metadata(sentence_id, engine):
    """Retrieve article metadata given one sentence id.

    Parameters
    ----------
    sentence_id: int
        Sentence id for which need to retrieve the article metadat.
    engine: sqlalchemy.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article: pd.DataFrame
        DataFrame containing the article metadata from
        which the sentence is coming.
    """
    sql_query = f"""SELECT * 
                    FROM articles 
                    WHERE article_id = 
                        (SELECT article_id 
                        FROM sentences
                        WHERE sentence_id = {sentence_id})"""
    article = pd.read_sql(sql_query, engine)
    return article


def retrieve_article(sentence_id, engine):
    """Retrieve article given one sentence id.

    Parameters
    ----------
    sentence_id: int
        Sentence id for which need to retrieve the article metadat.
    engine: sqlalchemy.Engine
        SQLAlchemy Engine connected to the database.

    Returns
    -------
    article: str
        Article containing the sentence of the given sentence_id.
    """
    sql_query = f"""SELECT text 
                    FROM sentences
                    WHERE article_id = 
                        (SELECT article_id 
                        FROM sentences 
                        WHERE sentence_id = {sentence_id})
                    ORDER BY paragraph_pos_in_article ASC, 
                    sentence_pos_in_paragraph ASC"""

    all_sentences = pd.read_sql(sql_query, engine)['text'].to_list()
    article = ' '.join(all_sentences)
    return article


def retrieve_paragraph(sentence_id, engine):
    """Retrieve paragraph given one sentence id.

    Parameters
    ----------
    sentence_id: int
        Sentence id for which need to retrieve the article metadat.
    engine: sqlalchemy.Engine
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
    paragraph = ' '.join(all_sentences)
    return paragraph


def find_paragraph(sentence_id, db_cnxn):
    """Find the paragraph corresponding to the given sentence.

    Parameters
    ----------
    sentence_id : int
        The identifier of the given sentence
    db_cnxn: SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database

    Returns
    -------
    paragraph : str
        The paragraph containing `sentence`
    """
    sql_query = f'SELECT paragraph_id FROM sentences WHERE sentence_id = {sentence_id}'
    paragraph_id = pd.read_sql(sql_query, db_cnxn).iloc[0]['paragraph_id']
    sql_query = f'SELECT text FROM paragraphs WHERE paragraph_id = {paragraph_id}'
    paragraph = pd.read_sql(sql_query, db_cnxn).iloc[0]['text']

    return paragraph


def get_shas_from_ids(articles_ids, db_cnxn):
    """Find articles SHA given article IDs.

    Parameters
    ----------
    articles_ids : list
        A list of strings representing article IDs.
    db_cnxn : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        A SQL database for querying the SHAs. Should contain
        a table named "article_id_2_sha".

    Returns
    -------
    results : list
        A list of sentence SHAs.
    """
    all_ids_str = ', '.join([f"'{id_}'" for id_ in articles_ids])
    sql_query = f"SELECT sha FROM article_id_2_sha WHERE article_id IN ({all_ids_str})"
    results = pd.read_sql(sql_query, db_cnxn)['sha'].tolist()

    return results


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
        condition = f"date BETWEEN '{date_from}-01-01' and '{date_to}-12-31'"

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
    def get_restrict_to_tag_condition(tag='has_covid19_tag'):
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

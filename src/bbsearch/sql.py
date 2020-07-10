"""
SQL Related functions.

whatever
"""
import pandas as pd


def get_paragraph_ids(article_ids, db_cnxn):
    """Given a list of article ids find all the corresponding paragraph ids.

    Parameters
    ----------
    article_ids : list
        List of article ids. Note that they are the primary keys in the `articles` table.

    db_cnxn : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database.

    Returns
    -------
    pd.Series
        The unique index represents the paragraph ids and the values represent the article ids.
    """
    article_ids_joined = ','.join(f"\"{id_}\"" for id_ in set(article_ids))

    sql_query = f"""
    SELECT article_id, paragraph_id
    FROM (
             SELECT paragraph_id, sha
             FROM paragraphs
             WHERE sha IN (
                 SELECT sha
                 FROM article_id_2_sha
                 WHERE article_id IN ({article_ids_joined})
             )
         ) p
             INNER JOIN
         article_id_2_sha a
         ON a.sha = p.sha;
    """
    results = pd.read_sql(sql_query, db_cnxn)

    return pd.Series(results['article_id'].tolist(), index=results['paragraph_id'])


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

    # paragraph_id = db.execute('SELECT paragraph_id FROM sentences WHERE sentence_id = ? ', [sentence_id]).fetchone()[0]
    # paragraph = db.execute('SELECT text FROM paragraphs WHERE paragraph_id = ?', [paragraph_id]).fetchone()[0]

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
    # results = db.execute(sql_query).fetchall()
    # results = [sha for (sha,) in results]

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
    # results = db.execute(sql_query).fetchall()
    # results = [id_ for (id_,) in results]

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

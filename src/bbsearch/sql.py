"""
SQL Related functions.

whatever
"""


def find_sentence_ids(restricted_article_ids, db):
    """Find sentence IDs given article SHAs.

    Parameters
    ----------
    restricted_article_ids : list
        A list of strings representing article SHAs.
    db : sqlite3.Connection
        A SQL database for querying the sentence IDs. Should contain
        a table named "sections".

    Returns
    -------
    results : list
        A list of sentence IDs.
    """
    all_shas_str = ', '.join([f"'{sha}'" for sha in restricted_article_ids])
    sql_query = f"SELECT Id FROM sections WHERE Article IN ({all_shas_str})"
    results = db.execute(sql_query).fetchall()
    results = [sha for (sha,) in results]

    return results


def find_article_ids_from_conditions(conditions, db):
    """Find article IDs given a number of search conditions.

    Parameters
    ----------
    conditions : list
        A list strings representing SQL query conditions. They should be
        formatted so that they can be used in an SQL WHERE statement,
        for example:
            SELECT * FROM articles WHERE <condition_1> and <condition_2>"
    db : sqlite3.Connection
        A SQL database for querying the article IDs. Should contain
        a table named "articles".

    Returns
    -------
    results : list
        A list of article IDs (=SHAs) represented as strings.
    """
    condition = ' and '.join(conditions)
    sql_query = f"SELECT Id FROM articles WHERE {condition}"
    results = db.execute(sql_query).fetchall()
    results = [sha for (sha,) in results]

    return results


def get_ids_by_condition(conditions, table, db):
    """Find entry IDs given a number of search conditions.

    Parameters
    ----------
    conditions : list
        A list strings representing SQL query conditions. They should be
        formatted so that they can be used in an SQL WHERE statement,
        for example:
            SELECT * FROM {table} WHERE <condition_1> and <condition_2>"
    table : str
        The name of the table in `db`.
    db : sqlite3.Connection
        A SQL database for querying the article IDs. Should contain
        a table named "articles".

    Returns
    -------
    results : list
        A list of article IDs (=SHAs) represented as strings.
    """
    condition = ' and '.join(conditions)
    sql_query = f"SELECT Id FROM {table} WHERE {condition}"
    results = db.execute(sql_query).fetchall()
    results = [id_ for (id_,) in results]

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
        condition = f"Published BETWEEN '{date_from}-01-01' and '{date_to}-12-31'"

        return condition

    @staticmethod
    def get_has_journal_condition():
        """Construct a has-journal condition.

        Returns
        -------
        condition : str
            The SQL condition
        """
        condition = "Publication IS NOT NULL"

        return condition


class SentenceConditioner:
    """Sentence conditioner."""

    @staticmethod
    def get_restrict_to_tag_condition(tag="COVID-19"):
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
        condition = f"Tags IS '{tag}'"
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
        condition = f"Text NOT LIKE '%{word}%'"
        return condition

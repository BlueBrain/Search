"""
SQL Related functions.

whatever
"""

def get_shas_from_ids(articles_ids, db):
    """Find articles SHA given given article IDs.

    Parameters
    ----------
    articles_ids : list
        A list of strings representing article IDs.
    db : sqlite3.Connection
        A SQL database for querying the SHAs. Should contain
        a table named "article_id_2_sha".

    Returns
    -------
    results : list
        A list of sentence SHAs.
    """
    all_ids_str = ', '.join([f"'{id_}'" for id_ in articles_ids])
    sql_query = f"SELECT sha FROM article_id_2_sha WHERE article_id IN ({all_ids_str})"
    results = db.execute(sql_query).fetchall()
    results = [sha for (sha,) in results]

    return results


def get_ids_from_shas(shas, db):
    """Find articles IDs given given article SHAs.

    Parameters
    ----------
    articles_ids : list
        A list of strings representing article SHAs.
    db : sqlite3.Connection
        A SQL database for querying the IDs. Should contain
        a table named "article_id_2_sha".

    Returns
    -------
    results : list
        A list of sentence IDs.
    """
    all_shas_str = ', '.join([f"'{sha}'" for sha in shas])
    sql_query = f"SELECT article_id FROM article_id_2_sha WHERE sha IN ({all_shas_str})"
    results = db.execute(sql_query).fetchall()
    results = [id_ for (id_,) in results]

    return results
    
    
def find_sentence_ids(article_shas, db):
    """Find sentence IDs given article SHAs.

    Parameters
    ----------
    restricted_article_ids : list
        A list of strings representing article SHAs.
    db : sqlite3.Connection
        A SQL database for querying the sentence IDs. Should contain
        a table named "sentences".

    Returns
    -------
    results : list
        A list of sentence IDs.
    """
    all_shas_str = ', '.join([f"'{sha}'" for sha in article_shas])
    sql_query = f"SELECT sentence_id FROM sentences WHERE sha IN ({all_shas_str})"
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
    sql_query = f"SELECT {table[:-1]}_id FROM {table} WHERE {condition}"
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
        condition = f"{tag} = True"
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

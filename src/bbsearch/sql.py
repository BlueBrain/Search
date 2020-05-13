"""
SQL Related functions.

whatever
"""
from pathlib import Path
import sqlite3

import pandas as pd

from bbsearch.utils import define_nlp, get_tag_and_sentences, update_covid19_tag, insert_into_sentences


class DatabaseCreation:
    """Create SQL database from a specified dataset."""

    def __init__(self,
                 data_path,
                 version,
                 saving_directory=None):
        """Creates SQL database object.

        Parameters
        ----------
        data_path: pathlib.Path
            Directory to the dataset where metadata.csv and all jsons file are located.
        version: str
            Version of the database created.
        saving_directory: pathlib.Path
            Directory where the database is going to be saved.
        """
        self.data_path = data_path
        if not Path(self.data_path).exists():
            raise NotADirectoryError(f'The data directory {self.data_path} does not exit')

        self.version = version

        self.saving_directory = saving_directory or Path.cwd()
        if not Path(self.saving_directory).exists():
            raise NotADirectoryError(f'The saving directory {self.saving_directory} does not exit')

        self.filename = self.saving_directory / f'cord19_{self.version}.db'

        self.metadata = pd.read_csv(self.data_path / 'metadata.csv')

    def construct(self):
        """Constructs the database."""

        self._rename_columns()
        self._schema_creation()
        self._article_id_to_sha_table()
        self._articles_table()
        self._sentences_table()

    def _schema_creation(self):
        """Creation of the schemas of the different tables in the database. """
        if self.filename.exists():
            raise ValueError(f'The version {self.version} of the database already exists')
        else:
            with sqlite3.connect(str(self.filename)) as db:
                db.execute(
                    """CREATE TABLE IF NOT EXISTS article_id_2_sha
                    (
                        article_id TEXT,
                        sha TEXT
                    );
                    """)
                db.execute(
                    """CREATE TABLE IF NOT EXISTS articles
                    (
                        article_id TEXT PRIMARY KEY,
                        publisher TEXT,
                        title TEXT,
                        doi TEXT,
                        pmc_id TEXT,
                        pm_id INTEGER,
                        licence TEXT,
                        abstract TEXT,
                        date DATETIME,
                        authors TEXT,
                        journal TEXT,
                        microsoft_id INTEGER,
                        covidence_id TEXT,
                        has_pdf_parse BOOLEAN,
                        has_pmc_xml_parse BOOLEAN,
                        has_covid19_tag BOOLEAN DEFAULT False,
                        fulltext_directory TEXT,
                        url TEXT
                    );
                    """)
                db.execute(
                    """CREATE TABLE sentences
                    (
                        sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sha TEXT,
                        section_name TEXT,
                        text TEXT,
                        FOREIGN KEY(sha) REFERENCES article_id_2_sha(sha)
                    );
                    """)

    def _rename_columns(self):
        """Renames the columns of the dataframe to follow the SQL database schema. """
        df = self.metadata
        df.rename(columns={
            'cord_uid': 'article_id',
            'sha': 'sha',
            'source_x': 'publisher',
            'title': 'title',
            'doi': 'doi',
            'pmcid': 'pmc_id',
            'pubmed_id': 'pm_id',
            'license': 'licence',
            'abstract': 'abstract',
            'publish_time': 'date',
            'authors': 'authors',
            'journal': 'journal',
            'Microsoft Academic Paper ID': 'microsoft_id',
            'WHO #Covidence': 'covidence_id',
            'has_pdf_parse': 'has_pdf_parse',
            'has_pmc_xml_parse': 'has_pmc_xml_parse',
            'full_text_file': 'fulltext_directory',
            'url': 'url'}, inplace=True)

    def _articles_table(self):
        """Fills the Article Table thanks to 'metadata.csv'.

        Notes
        -----
        The Dataframe self.metadata is modified in this method.
        The article_id_to_sha should be created before calling this method.
        """
        df = self.metadata.copy()
        df = df[df.columns[~df.columns.isin(['sha'])]]
        df.drop_duplicates('article_id', keep='first', inplace=True)
        with sqlite3.connect(str(self.filename)) as db:
            df.to_sql(name='articles', con=db, index=False, if_exists='append')

    def _article_id_to_sha_table(self):
        """Fills the article_id_to_sha table thanks to 'metadata.csv'. '"""
        df = self.metadata[['article_id', 'sha']]
        df = df.set_index(['article_id']).apply(lambda x: x.str.split('; ').explode()).reset_index()
        with sqlite3.connect(str(self.filename)) as db:
            df.to_sql(name='article_id_2_sha', con=db, index=False, if_exists='append')

    def _sentences_table(self):
        """Fills the sentences table thanks to all the json files. """
        nlp = define_nlp()
        with sqlite3.connect(str(self.filename)) as db:

            cur = db.cursor()
            for (article_id,) in cur.execute('SELECT article_id FROM articles'):
                tag, sentences = get_tag_and_sentences(db, nlp, self.data_path, article_id)
                update_covid19_tag(db, article_id, tag)
                insert_into_sentences(db, sentences)

            db.commit()


def get_shas_from_ids(articles_ids, db):
    """Find articles SHA given article IDs.

    Parameters
    ----------
    articles_ids : list
        A list of strings representing article IDs.
    db : sqlite3.Cursor
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


def get_ids_by_condition(conditions, table, db):
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
    db : sqlite3.Cursor
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

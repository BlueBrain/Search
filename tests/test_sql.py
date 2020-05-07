"""Collection of tests regarding the Database creation. """
import os
from pathlib import Path
import sqlite3

import pandas as pd

from bbsearch.sql import DatabaseCreation

VERSION = 'test'


class TestDatabaseCreation:
    """Tests the creation of the Database"""

    @classmethod
    def setup_class(cls):
        db = DatabaseCreation(data_path=Path('tests/data/'),
                              cord_path=Path('tests/data/CORD19_samples/'),
                              version=VERSION)
        db.construct()
        cls.database_path = Path(f'cord19_{VERSION}.db')

    @classmethod
    def teardown_class(cls):
        os.remove(str(cls.database_path))

    def test_database_creation(self):
        """Tests that a database has been created. """
        assert self.database_path.exists()

    def test_tables(self):
        """Tests that the three tables expected has been created. """
        with sqlite3.connect(self.database_path) as db:
            curs = db.cursor()
            tables_names = curs.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            tables_names = [table_name for (table_name, ) in tables_names]
            assert 'article_id_2_sha' in tables_names
            assert 'sentences' in tables_names
            assert 'articles' in tables_names

    def test_tables_content(self):
        """Tests that the tables are correctly filled. """
        with sqlite3.connect(self.database_path) as db:
            df = pd.read_sql("SELECT * FROM articles", db)
            assert df.shape[0] == 11
            df2 = pd.read_sql("SELECT * FROM article_id_2_sha", db)
            assert df2.shape[0] == 11
            df3 = df2[df2['sha'].notnull()]
            assert df3.shape[0] == 6
            df4 = pd.read_sql("SELECT DISTINCT sha FROM sentences WHERE sha is NOT NULL", db)
            assert df4.shape[0] == 6

    def test_tables_columns(self):
        """Tests that the tables columns are the ones expected. """
        with sqlite3.connect(self.database_path) as db:
            columns_expected = {"article_id", "publisher", "title", "doi", "pmc_id", "pm_id", "licence", "abstract",
                                "date", "authors", "journal", "microsoft_id", "covidence_id", "has_pdf_parse",
                                "has_pmc_xml_parse", "has_covid19_tag", "fulltext_directory", "url"}
            articles_columns = set(pd.read_sql("SELECT * FROM articles LIMIT 1", db).columns)
            assert columns_expected == articles_columns
            article_id_to_sha_expected = {"article_id", "sha"}
            article_id_to_sha_columns = set(pd.read_sql("SELECT * FROM article_id_2_sha LIMIT 1", db).columns)
            assert article_id_to_sha_expected == article_id_to_sha_columns
            sentences_expected = {"sentence_id", "sha", "section_name", "text"}
            sentences_columns = set(pd.read_sql("SELECT * FROM sentences LIMIT 1", db).columns)
            assert sentences_expected == sentences_columns

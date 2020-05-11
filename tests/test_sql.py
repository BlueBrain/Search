"""Collection of tests regarding the Database creation. """
import os
from pathlib import Path
import pytest
import sqlite3

import pandas as pd

from bbsearch.sql import DatabaseCreation


@pytest.fixture(scope='module')
def real_db_cnxn(tmp_path_factory, jsons_path):
    version = 'test'
    saving_directory = tmp_path_factory.mktemp('real_db', numbered=False)
    db = DatabaseCreation(data_path=jsons_path,
                          saving_directory=saving_directory,
                          version=version)
    db.construct()

    database_path = saving_directory / f'cord19_{version}.db'
    cnxn = sqlite3.connect(str(database_path))

    return cnxn


class TestDatabaseCreation:
    """Tests the creation of the Database"""

    def test_tables(self, real_db_cnxn):
        """Tests that the three tables expected has been created. """
        curs = real_db_cnxn.cursor()
        tables_names = curs.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        tables_names = [table_name for (table_name, ) in tables_names]
        assert 'article_id_2_sha' in tables_names
        assert 'sentences' in tables_names
        assert 'articles' in tables_names

    def test_tables_content(self, real_db_cnxn):
        """Tests that the tables are correctly filled. """
        df = pd.read_sql("SELECT * FROM articles", real_db_cnxn)
        assert df.shape[0] == 11
        df2 = pd.read_sql("SELECT * FROM article_id_2_sha", real_db_cnxn)
        assert df2.shape[0] == 11
        df3 = df2[df2['sha'].notnull()]
        assert df3.shape[0] == 6
        df4 = pd.read_sql("SELECT DISTINCT sha FROM sentences WHERE sha is NOT NULL", real_db_cnxn)
        assert df4.shape[0] == 6

    def test_tables_columns(self, real_db_cnxn):
        """Tests that the tables columns are the ones expected. """
        columns_expected = {"article_id", "publisher", "title", "doi", "pmc_id", "pm_id", "licence", "abstract",
                            "date", "authors", "journal", "microsoft_id", "covidence_id", "has_pdf_parse",
                            "has_pmc_xml_parse", "has_covid19_tag", "fulltext_directory", "url"}
        articles_columns = set(pd.read_sql("SELECT * FROM articles LIMIT 1", real_db_cnxn).columns)
        assert columns_expected == articles_columns
        article_id_to_sha_expected = {"article_id", "sha"}
        article_id_to_sha_columns = set(pd.read_sql("SELECT * FROM article_id_2_sha LIMIT 1", real_db_cnxn).columns)
        assert article_id_to_sha_expected == article_id_to_sha_columns
        sentences_expected = {"sentence_id", "sha", "section_name", "text"}
        sentences_columns = set(pd.read_sql("SELECT * FROM sentences LIMIT 1", real_db_cnxn).columns)
        assert sentences_expected == sentences_columns

    def test_errors(self, tmpdir, jsons_path):
        fake_dir = Path(str(tmpdir)) / 'fake'
        with pytest.raises(NotADirectoryError):
            DatabaseCreation(data_path=fake_dir,
                             saving_directory=jsons_path,
                             version='v0')
        with pytest.raises(NotADirectoryError):
            DatabaseCreation(data_path=jsons_path,
                             saving_directory=fake_dir,
                             version='v0')
        with pytest.raises(ValueError):
            version = 'test'
            saving_directory = Path(str(tmpdir))
            (saving_directory / f'cord19_{version}.db').touch()
            db = DatabaseCreation(data_path=jsons_path,
                                  saving_directory=saving_directory,
                                  version='test')
            db.construct()

    def test_real_equals_fake_db(self, real_db_cnxn, fake_db_cursor):
        """Tests that the schema of the fake database is always the same as the real one. """
        real_db_cursor = real_db_cnxn.cursor()
        all_tables = ['articles', 'article_id_2_sha', 'sentences']
        for table_name in all_tables:
            fake_db_schema = fake_db_cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            real_db_schema = real_db_cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            assert real_db_schema
            assert fake_db_schema
            fake_db_set = set(map(lambda x: x[1:3], fake_db_schema))
            real_db_set = set(map(lambda x: x[1:3], real_db_schema))
            assert fake_db_set == real_db_set

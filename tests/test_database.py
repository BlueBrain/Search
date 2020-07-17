from pathlib import Path
import pytest

import pandas as pd
import sqlalchemy

from bbsearch.database import CORD19DatabaseCreation


@pytest.fixture(scope='module')
def real_sqlalchemy_engine(tmp_path_factory, jsons_path):
    version = 'test'
    saving_directory = tmp_path_factory.mktemp('real_db', numbered=False)
    Path(f'{saving_directory}/cord19_{version}.db').touch()

    engine = sqlalchemy.create_engine(f'sqlite:///{saving_directory}/cord19_{version}.db')
    db = CORD19DatabaseCreation(data_path=jsons_path,
                                engine=engine)
    db.construct()

    return engine


class TestDatabaseCreation:
    """Tests the creation of the Database"""

    def test_tables(self, fake_sqlalchemy_engine):
        """Tests that the three tables expected has been created. """
        inspector = sqlalchemy.inspect(fake_sqlalchemy_engine)
        tables_names = [table_name for table_name in inspector.get_table_names()]
        assert 'sentences' in tables_names
        assert 'articles' in tables_names

    def test_tables_content(self, fake_sqlalchemy_engine):
        """Tests that the tables are correctly filled. """
        df = pd.read_sql("SELECT * FROM articles", fake_sqlalchemy_engine)
        assert df.shape[0] == 4
        df1 = pd.read_sql("SELECT DISTINCT article_id FROM sentences", fake_sqlalchemy_engine)
        assert df1.shape[0] == 4

    def test_tables_columns(self, fake_sqlalchemy_engine):
        """Tests that the tables columns are the ones expected. """
        columns_expected = {"article_id", "cord_uid", "sha", "source_x", "title", "doi", "pmcid",
                            "pubmed_id", "license", "abstract", "publish_time", "authors", "journal",
                            "mag_id", "arxiv_id", "pdf_json_files",
                            "pmc_json_files", "who_covidence_id", "s2_id", "url"}
        articles_columns = set(pd.read_sql("SELECT * FROM articles LIMIT 1",
                                           fake_sqlalchemy_engine).columns)
        assert columns_expected == articles_columns
        sentences_expected = {"sentence_id", "article_id", "section_name",
                              "text", "paragraph_pos_in_article", "sentence_pos_in_paragraph"}
        sentences_columns = set(pd.read_sql("SELECT * FROM sentences LIMIT 1",
                                            fake_sqlalchemy_engine).columns)
        assert sentences_expected == sentences_columns

    def test_errors(self, tmpdir, jsons_path):
        fake_dir = Path(str(tmpdir)) / 'fake'
        Path(f'{tmpdir}/cord19_test.db').touch()
        engine = sqlalchemy.create_engine(f'sqlite:///{tmpdir}/cord19_test.db')

        with pytest.raises(NotADirectoryError):
            CORD19DatabaseCreation(data_path=fake_dir,
                                   engine=engine)
        with pytest.raises(ValueError):
            db = CORD19DatabaseCreation(data_path=jsons_path,
                                        engine=engine)
            db.construct()
            db.construct()

    def test_real_equals_fake_db(self, real_sqlalchemy_engine, fake_sqlalchemy_engine):
        """Tests that the schema of the fake database is always the same as the real one. """
        inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        real_tables_names = {table_name for table_name in inspector.get_table_names()}
        fake_tables_names = fake_sqlalchemy_engine.execute("SELECT name FROM "
                                                           "sqlite_master WHERE type='table';").fetchall()
        fake_tables_names = {table_name for (table_name, ) in fake_tables_names}
        assert real_tables_names
        assert real_tables_names == fake_tables_names

        for table_name in real_tables_names:
            fake_db_schema = fake_sqlalchemy_engine.execute(f"PRAGMA table_info({table_name})").fetchall()
            real_db_schema = real_sqlalchemy_engine.execute(f"PRAGMA table_info({table_name})")
            assert real_db_schema
            assert fake_db_schema
            fake_db_set = set(map(lambda x: x[1:3], fake_db_schema))
            real_db_set = set(map(lambda x: x[1:3], real_db_schema))
            assert fake_db_set == real_db_set

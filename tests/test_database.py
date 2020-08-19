from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import sqlalchemy

from bbsearch.database import CORD19DatabaseCreation


@pytest.fixture()
def real_sqlalchemy_engine(jsons_path, monkeypatch, model_entities, tmpdir):

    fake_load = Mock()
    fake_load.return_value = model_entities

    monkeypatch.setattr('bbsearch.database.spacy.load', fake_load)

    version = 'test'
    Path(f'{tmpdir}/cord19_{version}.db').touch()

    engine = sqlalchemy.create_engine(f'sqlite:///{tmpdir}/cord19_{version}.db')
    db = CORD19DatabaseCreation(data_path=jsons_path,
                                engine=engine)
    db.construct()
    fake_load.assert_called_once()

    return engine


class TestDatabaseCreation:
    """Tests the creation of the Database"""

    def test_database_content(self, real_sqlalchemy_engine):
        """Tests that the two tables expected has been created. """
        inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        tables_names = [table_name for table_name in inspector.get_table_names()]
        assert 'sentences' in tables_names
        assert 'articles' in tables_names

        df = pd.read_sql("SELECT * FROM articles", real_sqlalchemy_engine)
        assert df.shape[0] == 4
        df1 = pd.read_sql("SELECT DISTINCT article_id FROM sentences", real_sqlalchemy_engine)
        assert df1.shape[0] == 4

        columns_expected = {"article_id", "cord_uid", "sha", "source_x", "title", "doi", "pmcid",
                            "pubmed_id", "license", "abstract", "publish_time", "authors", "journal",
                            "mag_id", "arxiv_id", "pdf_json_files",
                            "pmc_json_files", "who_covidence_id", "s2_id", "url"}
        articles_columns = set(pd.read_sql("SELECT * FROM articles LIMIT 1",
                                           real_sqlalchemy_engine).columns)
        assert columns_expected == articles_columns
        sentences_expected = {"sentence_id", "article_id", "section_name",
                              "text", "paragraph_pos_in_article", "sentence_pos_in_paragraph"}
        sentences_columns = set(pd.read_sql("SELECT * FROM sentences LIMIT 1",
                                            real_sqlalchemy_engine).columns)
        assert sentences_expected == sentences_columns

        inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        indexes_articles = inspector.get_indexes('articles')
        indexes_sentences = inspector.get_indexes('sentences')

        assert not indexes_articles
        assert len(indexes_sentences) == 1
        assert indexes_sentences[0]['column_names'][0] == 'article_id'

        duplicates_query = """SELECT COUNT(article_id || ':' ||
                                            paragraph_pos_in_article || ':' ||
                                            sentence_pos_in_paragraph) c,
                      article_id, paragraph_pos_in_article, sentence_pos_in_paragraph
                      FROM sentences
                      GROUP BY article_id, paragraph_pos_in_article, sentence_pos_in_paragraph
                      HAVING c > 1; """
        duplicates_df = pd.read_sql(duplicates_query, real_sqlalchemy_engine)
        assert len(duplicates_df) == 0

    def test_errors(self, tmpdir, jsons_path, monkeypatch, model_entities):

        fake_load = Mock()
        fake_load.return_value = model_entities

        monkeypatch.setattr('bbsearch.database.spacy.load', fake_load)

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
        real_tables_names = {table_name for table_name in real_sqlalchemy_engine.table_names()}
        fake_tables_names = {table_name for table_name in fake_sqlalchemy_engine.table_names()}

        assert real_tables_names == {'articles', 'sentences'}
        assert real_tables_names.issubset(fake_tables_names)

        real_inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        fake_inspector = sqlalchemy.inspect(fake_sqlalchemy_engine)

        for table_name in real_tables_names:
            real_columns = real_inspector.get_columns(table_name)
            fake_columns = fake_inspector.get_columns(table_name)
            assert real_columns
            assert fake_columns
            for x, y in zip(real_columns, fake_columns):
                assert str(x['type']) == str(y['type'])
                assert x['name'] == y['name']

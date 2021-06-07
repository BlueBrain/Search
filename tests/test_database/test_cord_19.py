"""Tests covering the creation of the CORD-19 database."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import sqlalchemy

from bluesearch.database import CORD19DatabaseCreation, mark_bad_sentences


@pytest.fixture()
def real_sqlalchemy_engine(
    jsons_path, monkeypatch, model_entities, fake_sqlalchemy_engine, tmpdir
):

    fake_load = Mock()
    fake_load.return_value = model_entities

    monkeypatch.setattr("bluesearch.database.cord_19.load_spacy_model", fake_load)

    version = "test"
    if fake_sqlalchemy_engine.url.drivername.startswith("mysql"):
        fake_sqlalchemy_engine.execute("drop database if exists real_test")
        fake_sqlalchemy_engine.execute("create database real_test")
        fake_url = fake_sqlalchemy_engine.url
        url = (
            f"{fake_url.drivername}://{fake_url.username}:{fake_url.password}@"
            f"{fake_url.host}:{fake_url.port}/"
        )
        engine = sqlalchemy.create_engine(f"{url}real_test")
    else:
        Path(f"{tmpdir}/cord19_{version}.db").touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{tmpdir}/cord19_{version}.db")

    db = CORD19DatabaseCreation(data_path=jsons_path, engine=engine)
    db.construct()
    fake_load.assert_called_once()

    return engine


def test_mark_bad_sentences(fake_sqlalchemy_engine):
    # Create a fake database
    df = pd.read_sql("select * from sentences", fake_sqlalchemy_engine)
    short_sentence = "hello"
    long_sentence = "a" * 3000
    latex_sentence = "\\documentclass{article}"

    # Test without bad sentences
    df["is_bad"] = 0
    df.to_sql("sentences_new", fake_sqlalchemy_engine, index=False)
    mark_bad_sentences(fake_sqlalchemy_engine, "sentences_new")
    df = pd.read_sql("select * from sentences_new", fake_sqlalchemy_engine)
    is_bad_nothing = df["is_bad"].copy()

    # Test with bad sentences
    df["is_bad"] = 0
    df.loc[0, "text"] = short_sentence
    df.loc[1, "text"] = long_sentence
    df.loc[2, "text"] = latex_sentence
    df.to_sql("sentences_new", fake_sqlalchemy_engine, index=False, if_exists="replace")

    # Mark bad sentences
    mark_bad_sentences(fake_sqlalchemy_engine, "sentences_new")

    df = pd.read_sql("select * from sentences_new", fake_sqlalchemy_engine)
    is_bad_3 = df["is_bad"].copy()

    with fake_sqlalchemy_engine.begin() as connection:
        connection.execute("drop table sentences_new")

    assert is_bad_nothing.sum() == 0
    assert is_bad_3.sum() == 3
    assert all(is_bad_3[:3])


class TestDatabaseCreation:
    """Tests the creation of the Database"""

    def test_database_content(self, real_sqlalchemy_engine):
        """Tests that the two tables expected has been created."""
        inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        tables_names = list(inspector.get_table_names())
        assert "sentences" in tables_names
        assert "articles" in tables_names

        df = pd.read_sql("SELECT * FROM articles", real_sqlalchemy_engine)
        assert df.shape[0] == 4
        df1 = pd.read_sql(
            "SELECT DISTINCT article_id FROM sentences", real_sqlalchemy_engine
        )
        assert df1.shape[0] == 4

        columns_expected = {
            "article_id",
            "cord_uid",
            "sha",
            "source_x",
            "title",
            "doi",
            "pmcid",
            "pubmed_id",
            "license",
            "abstract",
            "publish_time",
            "authors",
            "journal",
            "mag_id",
            "arxiv_id",
            "pdf_json_files",
            "pmc_json_files",
            "who_covidence_id",
            "s2_id",
            "url",
            "is_english",
        }
        articles_columns = set(
            pd.read_sql(
                "SELECT * FROM articles LIMIT 1", real_sqlalchemy_engine
            ).columns
        )
        assert columns_expected == articles_columns
        sentences_expected = {
            "sentence_id",
            "article_id",
            "section_name",
            "text",
            "paragraph_pos_in_article",
            "sentence_pos_in_paragraph",
            "is_bad",
        }
        sentences_columns = set(
            pd.read_sql(
                "SELECT * FROM sentences LIMIT 1", real_sqlalchemy_engine
            ).columns
        )
        assert sentences_expected == sentences_columns

        inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        indexes_articles = inspector.get_indexes("articles")
        indexes_sentences = inspector.get_indexes("sentences")

        assert not indexes_articles
        if real_sqlalchemy_engine.url.drivername.startswith("mysql"):
            assert (
                len(indexes_sentences) == 4
            )  # article_id, FULLTEXT index, unique_identifier
            for index in indexes_sentences:
                assert index["name"] in {
                    "sentence_unique_identifier",
                    "article_id_index",
                    "is_bad_article_id_index",
                    "fulltext_text",
                }
        else:
            assert len(indexes_sentences) == 2
            for index in indexes_sentences:
                assert index["name"] in {
                    "article_id_index",
                    "is_bad_article_id_index",
                }

        duplicates_query = """
        SELECT COUNT(
            article_id || ':' ||
            paragraph_pos_in_article || ':' ||
            sentence_pos_in_paragraph
        ) c,
        article_id,
        paragraph_pos_in_article,
        sentence_pos_in_paragraph
        FROM sentences
        GROUP BY article_id, paragraph_pos_in_article, sentence_pos_in_paragraph
        HAVING c > 1;
        """
        duplicates_df = pd.read_sql(duplicates_query, real_sqlalchemy_engine)
        assert len(duplicates_df) == 0

    def test_errors(self, tmpdir, jsons_path, monkeypatch, model_entities):

        fake_load = Mock()
        fake_load.return_value = model_entities

        monkeypatch.setattr("bluesearch.database.cord_19.load_spacy_model", fake_load)

        fake_dir = Path(str(tmpdir)) / "fake"
        Path(f"{tmpdir}/cord19_test.db").touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{tmpdir}/cord19_test.db")

        with pytest.raises(NotADirectoryError):
            CORD19DatabaseCreation(data_path=fake_dir, engine=engine)
        with pytest.raises(ValueError):
            db = CORD19DatabaseCreation(data_path=jsons_path, engine=engine)
            db.construct()
            db.construct()

    def test_boolean_columns(self, real_sqlalchemy_engine):
        """Test that boolean columns only contain boolean."""
        is_english = pd.read_sql(
            """SELECT is_english FROM articles""", real_sqlalchemy_engine
        )
        is_bad = pd.read_sql("""SELECT is_bad FROM sentences""", real_sqlalchemy_engine)
        assert set(is_english["is_english"].unique()).issubset({0, 1})
        assert set(is_bad["is_bad"].unique()).issubset({0, 1})

    def test_real_equals_fake_db(self, real_sqlalchemy_engine, fake_sqlalchemy_engine):
        """Test real vs. fake database.

        Tests that the schema of the fake database is always the same as
        the real one.
        """
        real_tables_names = set(real_sqlalchemy_engine.table_names())
        fake_tables_names = set(fake_sqlalchemy_engine.table_names())

        assert real_tables_names == {"articles", "sentences"}
        assert real_tables_names.issubset(fake_tables_names)

        real_inspector = sqlalchemy.inspect(real_sqlalchemy_engine)
        fake_inspector = sqlalchemy.inspect(fake_sqlalchemy_engine)

        for table_name in real_tables_names:
            real_columns = real_inspector.get_columns(table_name)
            fake_columns = fake_inspector.get_columns(table_name)
            assert real_columns
            assert fake_columns
            assert len(real_columns) == len(fake_columns)
            for x, y in zip(real_columns, fake_columns):
                assert str(x["type"]) == str(y["type"])
                assert x["name"] == y["name"]

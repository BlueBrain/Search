"""Configuration of pytest."""

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

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import spacy
import sqlalchemy
from sqlalchemy.exc import OperationalError

import docker

ROOT_PATH = Path(__file__).resolve().parent.parent  # root of the repository
CORD19_SAMPLE_PATH = ROOT_PATH / "tests" / "data" / "cord19_v35"
CORD19_SAMPLE_ALL_JSON_PATHS = sorted(CORD19_SAMPLE_PATH.rglob("*.json"))


@pytest.fixture(scope="session")
def test_parameters(metadata_path, entity_types):
    """Parameters needed for the tests"""
    return {
        "n_articles": len(pd.read_csv(metadata_path)),
        "n_sections_per_article": 2,  # paragraph = section
        "n_sentences_per_section": 3,
        "n_entities_per_section": len(entity_types),
        "embedding_size": 2,
    }


def fill_db_data(engine, metadata_path, test_parameters, entity_types):
    metadata = sqlalchemy.MetaData()

    # Creation of the schema of the tables
    sqlalchemy.Table(
        "articles",
        metadata,
        sqlalchemy.Column(
            "article_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("cord_uid", sqlalchemy.String(8), nullable=False),
        sqlalchemy.Column("sha", sqlalchemy.Text()),
        sqlalchemy.Column("source_x", sqlalchemy.Text()),
        sqlalchemy.Column("title", sqlalchemy.Text()),
        sqlalchemy.Column("doi", sqlalchemy.Text()),
        sqlalchemy.Column("pmcid", sqlalchemy.Text()),
        sqlalchemy.Column("pubmed_id", sqlalchemy.Text()),
        sqlalchemy.Column("license", sqlalchemy.Text()),
        sqlalchemy.Column("abstract", sqlalchemy.Text()),
        sqlalchemy.Column("publish_time", sqlalchemy.Date()),
        sqlalchemy.Column("authors", sqlalchemy.Text()),
        sqlalchemy.Column("journal", sqlalchemy.Text()),
        sqlalchemy.Column("mag_id", sqlalchemy.Text()),
        sqlalchemy.Column("who_covidence_id", sqlalchemy.Text()),
        sqlalchemy.Column("arxiv_id", sqlalchemy.Text()),
        sqlalchemy.Column("pdf_json_files", sqlalchemy.Text()),
        sqlalchemy.Column("pmc_json_files", sqlalchemy.Text()),
        sqlalchemy.Column("url", sqlalchemy.Text()),
        sqlalchemy.Column("s2_id", sqlalchemy.Text()),
        sqlalchemy.Column("is_english", sqlalchemy.Boolean()),
    )

    sentences_table = sqlalchemy.Table(
        "sentences",
        metadata,
        sqlalchemy.Column(
            "sentence_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("section_name", sqlalchemy.Text()),
        sqlalchemy.Column(
            "article_id",
            sqlalchemy.Integer(),
            sqlalchemy.ForeignKey("articles.article_id"),
            nullable=False,
        ),
        sqlalchemy.Column("text", sqlalchemy.Text()),
        sqlalchemy.Column(
            "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
        ),
        sqlalchemy.Column(
            "sentence_pos_in_paragraph", sqlalchemy.Integer(), nullable=False
        ),
        sqlalchemy.Column("is_bad", sqlalchemy.Boolean(), server_default="0"),
        sqlalchemy.UniqueConstraint(
            "article_id",
            "paragraph_pos_in_article",
            "sentence_pos_in_paragraph",
            name="sentence_unique_identifier",
        ),
    )

    mining_cache = sqlalchemy.Table(
        "mining_cache",
        metadata,
        sqlalchemy.Column(
            "entity_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("entity", sqlalchemy.Text()),
        sqlalchemy.Column("entity_type", sqlalchemy.Text()),
        sqlalchemy.Column("property", sqlalchemy.Text()),
        sqlalchemy.Column("property_value", sqlalchemy.Text()),
        sqlalchemy.Column("property_type", sqlalchemy.Text()),
        sqlalchemy.Column("property_value_type", sqlalchemy.Text()),
        sqlalchemy.Column("paper_id", sqlalchemy.Text()),
        sqlalchemy.Column("start_char", sqlalchemy.Integer()),
        sqlalchemy.Column("end_char", sqlalchemy.Integer()),
        sqlalchemy.Column(
            "article_id",
            sqlalchemy.Integer(),
            sqlalchemy.ForeignKey("articles.article_id"),
            nullable=False,
        ),
        sqlalchemy.Column(
            "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
        ),
        sqlalchemy.Column("mining_model", sqlalchemy.Text()),
    )

    # Construction of the tables
    with engine.begin() as connection:
        metadata.create_all(connection)

    # Construction of the index 'article_id_index'
    sqlalchemy.Index("article_id_sentences_index", sentences_table.c.article_id).create(
        bind=engine
    )
    sqlalchemy.Index("article_id_mining_cache_index", mining_cache.c.article_id).create(
        bind=engine
    )

    # Population of the tables 'sentences' and 'articles'
    metadata_df = pd.read_csv(str(metadata_path))
    metadata_df["is_english"] = True
    metadata_df.index.name = "article_id"
    metadata_df.index += 1
    with engine.begin() as con:
        metadata_df.to_sql(name="articles", con=con, index=True, if_exists="append")

    temp_s = []
    for article_id in set(metadata_df[metadata_df.index.notna()].index.to_list()):
        for sec_ix in range(test_parameters["n_sections_per_article"]):
            for sen_ix in range(test_parameters["n_sentences_per_section"]):
                s = pd.Series(
                    {
                        "text": "I am a sentence {} in section {} "
                        "in article {}.".format(sen_ix, sec_ix, article_id),
                        "section_name": "section_{}".format(sec_ix),
                        "article_id": article_id,
                        "paragraph_pos_in_article": sec_ix,
                        "sentence_pos_in_paragraph": sen_ix,
                        "is_bad": sen_ix % 2,
                    }
                )
                temp_s.append(s)

    sentences_content = pd.DataFrame(temp_s)
    sentences_content.index.name = "sentence_id"
    sentences_content.index += 1
    with engine.begin() as con:
        sentences_content.to_sql(
            name="sentences", con=con, index=True, if_exists="append"
        )
    if engine.url.drivername.startswith("mysql"):
        engine.execute("""CREATE FULLTEXT INDEX fulltext_text ON sentences(text)""")

    # populate mining tables
    temp_m = []
    for article_id in set(metadata_df[metadata_df.index.notna()].index.to_list()):
        for sec_ix in range(test_parameters["n_sections_per_article"]):
            for ent_ix in range(test_parameters["n_entities_per_section"]):
                s = pd.Series(
                    {
                        "entity": f"entity_{ent_ix}",
                        "entity_type": entity_types[ent_ix],
                        "property": None,
                        "property_value": None,
                        "property_type": None,
                        "property_value_type": None,
                        "paper_id": f"{article_id}:whatever:{sec_ix}",
                        "start_char": ent_ix,
                        "end_char": ent_ix + 1,
                        "article_id": article_id,
                        "paragraph_pos_in_article": sec_ix,
                    }
                )
                temp_m.append(s)

    mining_content = pd.DataFrame(temp_m)
    mining_content.index.name = "entity_id"
    mining_content.index += 1
    with engine.begin() as con:
        mining_content.to_sql(
            name="mining_cache", con=con, index=True, if_exists="append"
        )


@pytest.fixture(scope="session", params=["sqlite", "mysql"])
def backend_database(request):
    """Check if different backends are available."""
    backend = request.param
    if backend == "mysql":
        # check docker daemon running
        try:
            client = docker.from_env()
            client.ping()

        except docker.errors.DockerException:
            pytest.skip()

    return backend


@pytest.fixture(scope="session")
def fake_sqlalchemy_engine(
    tmp_path_factory, metadata_path, test_parameters, backend_database, entity_types
):
    """Connection object (sqlite)."""
    if backend_database == "sqlite":
        db_path = tmp_path_factory.mktemp("db", numbered=False) / "cord19_test.db"
        Path(db_path).touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        fill_db_data(engine, metadata_path, test_parameters, entity_types)
        yield engine

    else:
        port_number = 22345
        client = docker.from_env()
        container = client.containers.run(
            image="mysql:latest",
            environment={"MYSQL_ROOT_PASSWORD": "my-secret-pw"},
            ports={"3306/tcp": port_number},
            detach=True,
            auto_remove=True,
        )

        max_waiting_time = 2 * 60
        start = time.perf_counter()

        while time.perf_counter() - start < max_waiting_time:
            try:
                engine = sqlalchemy.create_engine(
                    f"mysql+pymysql://root:my-secret-pw@127.0.0.1:{port_number}/"
                )
                # Container ready?
                engine.execute("show databases")
                break
            except OperationalError:
                # Container not ready, pause and then try again
                time.sleep(0.1)
                continue
        else:
            raise TimeoutError("Could not spawn the MySQL container.")

        engine.execute("create database test")
        engine.dispose()
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://root:my-secret-pw" f"@127.0.0.1:{port_number}/test"
        )
        fill_db_data(engine, metadata_path, test_parameters, entity_types)

        yield engine

        container.kill()


@pytest.fixture(scope="session")
def fake_sqlalchemy_cnxn(fake_sqlalchemy_engine):
    """Connection object (sqlite)."""
    sqlalchemy_cnxn = fake_sqlalchemy_engine.connect()
    return sqlalchemy_cnxn


@pytest.fixture(scope="session")
def jsons_path():
    """Path to a directory where jsons are stored."""
    jsons_path = CORD19_SAMPLE_PATH
    assert jsons_path.exists()

    return jsons_path


@pytest.fixture(
    params=CORD19_SAMPLE_ALL_JSON_PATHS,
    ids=(path.name for path in CORD19_SAMPLE_ALL_JSON_PATHS),
)
def real_json_file(request):
    with request.param.open() as fp:
        yield json.load(fp)


@pytest.fixture(scope="session")
def metadata_path():
    """Path to metadata.csv."""
    metadata_path = CORD19_SAMPLE_PATH / "metadata.csv"
    assert metadata_path.exists()

    return metadata_path


@pytest.fixture(scope="session")
def entity_types():
    """Entity types that can be used throughout tests."""
    return ["CHEMICAL", "ORGANISM"]


@pytest.fixture()
def spacy_model_path(tmpdir, entity_types):
    """Structure of files for entity models."""
    data_dir = Path(tmpdir) / "some" / "path"
    models_dir = data_dir / "models" / "ner_er"
    models_dir.mkdir(parents=True)

    for etype in entity_types:
        etype_model_dir = models_dir / f"model-{etype.lower()}"
        etype_model_dir.mkdir()
        meta_path = etype_model_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "labels": {
                        "ner": [
                            etype.upper(),
                        ]
                    }
                },
                f,
            )
    return data_dir


@pytest.fixture(scope="session")
def embeddings_h5_path(tmp_path_factory, fake_sqlalchemy_engine, test_parameters):
    random_state = 3
    np.random.seed(random_state)
    models = ["SBERT", "SBioBERT"]
    dim = test_parameters["embedding_size"]
    n_sentences = pd.read_sql(
        "SELECT COUNT(*) FROM sentences", fake_sqlalchemy_engine
    ).iloc[0, 0]
    if not (tmp_path_factory.getbasetemp() / "h5_embeddings").is_dir():
        file_path = (
            tmp_path_factory.mktemp("h5_embeddings", numbered=False) / "embeddings.h5"
        )

        with h5py.File(file_path, "w") as f:
            for model in models:
                dset = f.create_dataset(
                    f"{model}", (n_sentences + 1, dim), dtype="f4", fillvalue=np.nan
                )

                for i in range(1, n_sentences + 1):
                    dset[i] = np.random.random(dim).astype("float32")
    else:
        file_path = tmp_path_factory.getbasetemp() / "h5_embeddings" / "embeddings.h5"

    return file_path


@pytest.fixture(scope="session")
def model_entities():
    """Standard English spacy model.

    References
    ----------
    https://spacy.io/api/annotation#named-entities
    """
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="session")
def ner_annotations():
    csv_filename = (
        ROOT_PATH / "tests" / "data" / "mining" / "eval" / "ner_iob_sample.csv"
    )

    return {
        "bio": pd.read_csv(csv_filename),
        "sample": pd.DataFrame(
            data={
                "annotator_1": [
                    "B-a",
                    "B-a",
                    "B-b",
                    "B-a",
                    "O",
                    "B-a",
                    "I-a",
                    "O",
                    "B-b",
                    "I-b",
                    "O",
                    "O",
                    "B-d",
                    "B-b",
                ],
                "annotator_2": [
                    "B-c",
                    "B-c",
                    "I-c",
                    "B-c",
                    "O",
                    "B-c",
                    "O",
                    "B-b",
                    "I-b",
                    "I-b",
                    "B-c",
                    "I-c",
                    "B-c",
                    "B-b",
                ],
            }
        ),
        "sample_nested": pd.DataFrame(
            data={
                "annotator_1": ["O", "B-a", "O", "B-a", "I-a", "I-a", "O"],
                "annotator_2": ["O", "B-a", "O", "B-a", "B-a", "I-a", "B-a"],
                "text": [
                    "Either",
                    "influenza",
                    "or",
                    "Sars",
                    "Cov-2",
                    "infection",
                    "disease",
                ],
            }
        ),
    }


@pytest.fixture(scope="session")
def punctuation_annotations():
    files_location = ROOT_PATH / "tests" / "data" / "mining" / "eval"
    return {
        mode: pd.read_csv(files_location / f"iob_punctuation_{mode}.csv")
        for mode in ("before", "after")
    }


@pytest.fixture()
def mining_schema_df():
    schema_file = ROOT_PATH / "tests" / "data" / "mining" / "request" / "request.csv"
    df = pd.read_csv(schema_file)
    return df

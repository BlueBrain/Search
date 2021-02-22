"""Test for the mining server."""

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
import shutil
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from bluesearch.mining import SPECS
from bluesearch.server.mining_server import MiningServer
from bluesearch.utils import load_ee_models_library

TESTS_PATH = Path(__file__).resolve().parent.parent  # path to tests directory


@pytest.fixture
def mining_client(fake_sqlalchemy_engine, model_entities, monkeypatch, tmpdir):
    """Fixture to create a client for mining_server."""

    fake_load = Mock()
    fake_load.return_value = model_entities

    monkeypatch.setattr("bluesearch.server.mining_server.load_spacy_model", fake_load)

    # This is the original CSV file with 3 columns:
    # entity_type, model, entity_type_name
    models_libs = TESTS_PATH / "data" / "mining" / "request" / "ee_models_library.csv"

    # To load the library file we must use load_ee_models_library() as it
    # converts the column "model" to two new columns "model_id" and "model_path"
    # However, load_ee_models_library() assumes a particular directory
    # structure, that's why we copy the original CSV to a new place with the
    # correct directory structure.
    tmpdir = Path(tmpdir)
    new_csv_path = tmpdir / "pipelines" / "ner" / "ee_models_library.csv"
    new_csv_path.parent.mkdir(parents=True)
    shutil.copy(models_libs, new_csv_path)

    # Load the model library
    df_csv = load_ee_models_library(tmpdir)

    mining_server_app = MiningServer(
        models_libs={"ee": df_csv}, connection=fake_sqlalchemy_engine
    )
    mining_server_app.config["TESTING"] = True
    with mining_server_app.test_client() as client:
        yield client


class TestMiningServer:
    def test_mining_server_help(self, mining_client):
        response = mining_client.post("/help")
        assert response.json["name"] == "MiningServer"

    def test_mining_server_pipeline(self, mining_client):
        schema_file = TESTS_PATH / "data" / "mining" / "request" / "request.csv"
        with open(schema_file, "r") as f:
            schema_request = f.read()

        # Test a valid request
        request_json = {"text": "hello", "schema": schema_request}
        response = mining_client.post("/text", json=request_json)
        assert response.headers["Content-Type"] == "application/json"
        assert response.status_code == 200
        response_json = response.json
        missing_etypes = ["DISEASE", "CELL_TYPE", "PROTEIN", "ORGAN"]
        assert response_json["warnings"] == [
            f'No text mining model was found in the library for "{etype}".'
            for etype in missing_etypes
        ]
        assert (
            response_json["csv_extractions"].split("\n")[0]
            == "entity,entity_type,property,"
            "property_value,property_type,"
            "property_value_type,"
            "ontology_source,paper_id,"
            "start_char,end_char"
        )

        # Test request with a missing text
        request_json = {}
        response = mining_client.post("/text", json=request_json)
        assert response.status_code == 400
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": 'The request "text" is missing.'}

        # Test request with a missing schema
        request_json = {"text": "hello"}
        response = mining_client.post("/text", json=request_json)
        assert response.status_code == 400
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": 'The request "schema" is missing.'}

        # Test a non-JSON request
        response = mining_client.post("/text", data="text")
        assert response.status_code == 400
        assert response.json == {"error": "The request has to be a JSON object."}

    @pytest.mark.parametrize(
        "use_cache", [True, False], ids=["with_cache", "without_cache"]
    )
    def test_mining_server_database(self, mining_client, use_cache):
        schema_file = TESTS_PATH / "data" / "mining" / "request" / "request.csv"
        with open(schema_file, "r") as f:
            schema_request = f.read()

        # Test a request with a missing "identifiers" key
        response = mining_client.post("/database", json={})
        assert list(response.json.keys()) == ["error"]
        assert response.status_code == 400
        assert response.json == {"error": 'The request "identifiers" is missing.'}

        # Test a non-JSON request
        response = mining_client.post("/database", data="text")
        assert response.status_code == 400
        assert response.json == {"error": "The request has to be a JSON object."}

        identifiers = [(1, 0), (2, -1)]
        request_json = {
            "identifiers": identifiers,
            "schema": schema_request,
            "use_cache": use_cache,
        }
        response = mining_client.post("/database", json=request_json)
        response_json = response.json
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        missing_etypes = ["DISEASE", "CELL_TYPE", "PROTEIN", "ORGAN"]
        assert response_json["warnings"] == [
            f'No text mining model was found in the library for "{etype}".'
            for etype in missing_etypes
        ]
        assert (
            response_json["csv_extractions"].split("\n")[0]
            == "entity,entity_type,property,"
            "property_value,property_type,"
            "property_value_type,"
            "ontology_source,paper_id,"
            "start_char,end_char"
        )

    @pytest.mark.parametrize("debug", [True, False], ids=["debug", "specs"])
    def test_mining_cache_detailed(self, mining_client, test_parameters, debug):
        """Test exact count of found entities.

        This test assumes that the requested entity types are a superset of
        those present in the database (as defined in ee_models_library.csv).
        """
        schema_file = TESTS_PATH / "data" / "mining" / "request" / "request.csv"
        with open(schema_file, "r") as f:
            schema_request = f.read()

        identifiers = [(1, 0), (2, -1), (3, 1)]

        expected_length = 0
        for article_id, pos in identifiers:
            n_paragraphs = test_parameters["n_sections_per_article"] if pos == -1 else 1
            expected_length += test_parameters["n_entities_per_section"] * n_paragraphs

        request_json = {
            "identifiers": identifiers,
            "schema": schema_request,
            "use_cache": True,
            "debug": debug,
        }
        response = mining_client.post("/database", json=request_json)

        assert response.status_code == 200

        df = pd.read_csv(StringIO(response.json["csv_extractions"]))

        assert len(df) == expected_length
        assert debug ^ (df.columns.to_list() == SPECS)

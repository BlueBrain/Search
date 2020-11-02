"""Test for the mining server."""
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from bbsearch.mining import SPECS
from bbsearch.server.mining_server import MiningServer

TESTS_PATH = Path(__file__).resolve().parent.parent  # path to tests directory


@pytest.fixture
def mining_client(fake_sqlalchemy_engine, model_entities, monkeypatch):
    """Fixture to create a client for mining_server."""

    spacy_mock = Mock()
    spacy_mock.load.return_value = model_entities

    monkeypatch.setattr("bbsearch.server.mining_server.spacy", spacy_mock)

    models_libs = TESTS_PATH / "data" / "mining" / "request" / "ee_models_library.csv"
    mining_server_app = MiningServer(
        models_libs={"ee": str(models_libs)}, connection=fake_sqlalchemy_engine
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
        request_json = {}
        response = mining_client.post("/text", json=request_json)
        assert response.status_code == 400
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": 'The request "text" is missing.'}

        request_json = {"text": "hello"}
        response = mining_client.post("/text", json=request_json)
        assert response.status_code == 400
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": 'The request "schema" is missing.'}

        request_json = "text"
        response = mining_client.post("/text", data=request_json)
        assert response.status_code == 400
        assert response.json == {"error": "The request has to be a JSON object."}

    @pytest.mark.parametrize(
        "use_cache", [True, False], ids=["with_cache", "without_cache"]
    )
    def test_mining_server_database(self, mining_client, use_cache):
        schema_file = TESTS_PATH / "data" / "mining" / "request" / "request.csv"
        with open(schema_file, "r") as f:
            schema_request = f.read()
        request_json = {}
        response = mining_client.post("/database", json=request_json)
        assert list(response.json.keys()) == ["error"]
        assert response.status_code == 400
        assert response.json == {"error": 'The request "identifiers" is missing.'}

        request_json = "text"
        response = mining_client.post("/database", data=request_json)
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
    def test_mining_cache_detailed(
        self, mining_client, test_parameters, entity_types, debug
    ):
        """Test exact count of found entities.

        This test assumes that the requested entity types are a superset of those present
        in the database (as defined in ee_models_library.csv).
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

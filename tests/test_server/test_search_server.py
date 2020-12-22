from unittest.mock import Mock

import numpy as np
import pytest

from bbsearch.server.search_server import SearchServer
from bbsearch.utils import H5


@pytest.fixture
def search_client(
    monkeypatch, embeddings_h5_path, fake_sqlalchemy_engine, test_parameters
):
    """Fixture to create a client for mining_server."""

    bsv_model_inst = Mock()
    bsv_model_class = Mock()
    bsv_model_class.return_value = bsv_model_inst
    bsv_model_inst.preprocess.return_value = "hello"
    bsv_model_inst.embed.return_value = np.ones((test_parameters["embedding_size"],))

    sbiobert_model_inst = Mock()
    sbiobert_model_class = Mock()
    sbiobert_model_class.return_value = sbiobert_model_inst
    sbiobert_model_inst.preprocess.return_value = "hello"
    sbiobert_model_inst.embed.return_value = np.ones(
        (test_parameters["embedding_size"],)
    )

    monkeypatch.setattr("bbsearch.server.search_server.BSV", bsv_model_class)
    monkeypatch.setattr("bbsearch.server.search_server.SBioBERT", sbiobert_model_class)

    indices = H5.find_populated_rows(embeddings_h5_path, "BSV")

    search_server_app = SearchServer(
        trained_models_path="",
        embeddings_h5_path=embeddings_h5_path,
        indices=indices,
        connection=fake_sqlalchemy_engine,
        models=["BSV", "SBioBERT"],
    )
    search_server_app.config["TESTING"] = True
    with search_server_app.test_client() as client:
        yield client


class TestSearchServer:
    def test_search_server(self, search_client):
        # Test the help request
        response = search_client.post("/help")
        assert response.status_code == 200
        assert response.json["name"] == "SearchServer"

        # Test a valid JSON request
        k = 3
        request_json = {"which_model": "BSV", "k": k, "query_text": "hello"}
        response = search_client.post("/", json=request_json)
        assert response.status_code == 200
        json_response = response.json
        assert len(json_response["sentence_ids"]) == k
        assert len(json_response["similarities"]) == k

        # Test a non-JSON request
        response = search_client.post("/", data="data is not a json")
        assert response.status_code == 200
        json_response = response.json
        assert json_response["sentence_ids"] is None
        assert json_response["similarities"] is None

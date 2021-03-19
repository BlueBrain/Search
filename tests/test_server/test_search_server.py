"""Tests covering the search server."""

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

from unittest.mock import Mock

import numpy as np
import pytest

from bluesearch.server.search_server import SearchServer
from bluesearch.utils import H5


@pytest.fixture
def search_client(
    monkeypatch, embeddings_h5_path, fake_sqlalchemy_engine, test_parameters
):
    """Fixture to create a client for mining_server."""

    fake_embedding_model = Mock()
    fake_embedding_model.preprocess.return_value = "hello"
    fake_embedding_model.embed.return_value = np.ones(
        (test_parameters["embedding_size"],)
    )

    monkeypatch.setattr(
        "bluesearch.server.search_server.get_embedding_model",
        lambda *args, **kwargs: fake_embedding_model,
    )

    indices = H5.find_populated_rows(embeddings_h5_path, "SBioBERT")

    search_server_app = SearchServer(
        trained_models_path="",
        embeddings_h5_path=embeddings_h5_path,
        indices=indices,
        connection=fake_sqlalchemy_engine,
        models=["SBioBERT"],
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
        request_json = {"which_model": "SBioBERT", "k": k, "query_text": "hello"}
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

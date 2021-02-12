"""Tests covering embedding server"""

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

from bluesearch.server.embedding_server import EmbeddingServer


@pytest.fixture(scope="session")
def embedding_client():
    """Fixture to create a client for mining_server."""

    sbiobert = Mock()
    sbiobert.preprocess.return_value = "This is a dummy sentence"
    sbiobert.embed.return_value = np.ones((2,))
    embedding_models = {"sbiobert": sbiobert}

    embedding_server_app = EmbeddingServer(embedding_models=embedding_models)
    embedding_server_app.config["TESTING"] = True
    with embedding_server_app.test_client() as client:
        yield client


class TestEmbeddingServer:
    def test_embedding_server_help(self, embedding_client):
        response = embedding_client.post("/help")
        assert response.status_code == 200
        assert response.json["name"] == "EmbeddingServer"

    def test_embedding_server_welcome(self, embedding_client):
        response = embedding_client.get("/")
        assert response.status_code == 200
        response = embedding_client.post("/")
        assert response.status_code == 405

    def test_embedding_server_embed(self, embedding_client):
        request_json = {"model": "sbiobert", "text": "hello"}
        response = embedding_client.post("/v1/embed/json", json=request_json)
        assert response.status_code == 200

        request_json = {"model": "sbiobert"}
        response = embedding_client.post("/v1/embed/json", json=request_json)
        assert response.status_code == 400

        request_json = {"model": "sbiobert", "text": "hello"}
        response = embedding_client.post("/v1/embed/csv", json=request_json)
        assert response.status_code == 200

        request_json = {"model": "invalid_model", "text": "hello"}
        response = embedding_client.post("/v1/embed/csv", json=request_json)
        assert response.status_code == 400

        response = embedding_client.post("/v1/embed/csv", data="not json")
        assert response.status_code == 400

        response = embedding_client.post("/v1/embed/invalid_format", data="not json")
        assert response.status_code == 400

from flask import Flask
import pytest

from bbsearch.embedding_models import SBioBERT
from bbsearch.server.embedding_server import EmbeddingServer


@pytest.fixture
def embedding_client(monkeypatch):
    """Fixture to create a client for mining_server."""

    sbiobert = SBioBERT()
    embedding_models = {'sbiobert': sbiobert}

    app = Flask("BBSearch Test Embedding Server")
    embedding_server = EmbeddingServer(app=app,
                                       embedding_models=embedding_models)
    embedding_server.app.config['TESTING'] = True
    with embedding_server.app.test_client() as client:
        yield client


class TestEmbeddingServer:

    def test_embedding_server_welcome(self, embedding_client):
        response = embedding_client.get('/')
        assert response.status_code == 200
        response = embedding_client.post('/')
        assert response.status_code == 405

    def test_embedding_server_embed(self, embedding_client):
        request_json = {'model': 'sbiobert',
                        'text': 'hello'}
        response = embedding_client.post('/v1/embed/json', json=request_json)
        assert response.status_code == 200

        request_json = {'model': 'sbiobert'}
        response = embedding_client.post('/v1/embed/json', json=request_json)
        assert response.status_code == 400

        request_json = {'model': 'sbiobert',
                        'text': 'hello'}
        response = embedding_client.post('/v1/embed/csv', json=request_json)
        assert response.status_code == 200

        request_json = {'model': 'invalid_model',
                        'text': 'hello'}
        response = embedding_client.post('/v1/embed/csv', json=request_json)
        assert response.status_code == 400

        request_json = 'not a json as request'
        response = embedding_client.post('/v1/embed/csv', data=request_json)
        assert response.status_code == 400

        request_json = 'not a json as request'
        response = embedding_client.post('/v1/embed/invalid_format', data=request_json)
        assert response.status_code == 400

"""Test for the mining server."""
from flask import Flask
import pytest
from unittest.mock import Mock, MagicMock

from bbsearch.mining.relation import ChemProt
from bbsearch.server.mining_server import MiningServer


@pytest.fixture
def mining_client(fake_db_cnxn, model_entities, monkeypatch):
    """Fixture to create a client for mining_server."""

    spacy_mock = Mock()
    spacy_mock.load.return_value = model_entities

    monkeypatch.setattr('bbsearch.server.mining_server.spacy', spacy_mock)

    app = Flask("BBSearch Test Mining Server")
    database_path = fake_db_cnxn.execute("""PRAGMA database_list""").fetchall()[0][2]

    mining_server = MiningServer(app=app,
                                 models_path='',
                                 database_path=database_path)
    mining_server.app.config['TESTING'] = True
    with mining_server.app.test_client() as client:
        yield client


class TestMiningServer:

    def test_mining_server_help(self, mining_client):
        response = mining_client.post('/help')
        assert response.json['name'] == 'MiningServer'

    def test_mining_server_pipeline(self, mining_client):
        request_json = {"text": 'hello'}
        response = mining_client.post('/text', json=request_json)
        assert response.headers['Content-Type'] == 'text/csv'
        assert response.data.decode('utf-8').split('\n')[0] == 'entity,entity_type,property,' \
                                                               'property_value,property_type,' \
                                                               'property_value_type,' \
                                                               'ontology_source,paper_id,' \
                                                               'start_char,end_char'
        request_json = {}
        response = mining_client.post('/text', json=request_json)
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": "The request text is missing."}
        request_json = "text"
        response = mining_client.post('/text', data=request_json)
        assert response.json == {"error": "The request has to be a JSON object."}

    def test_mining_server_database(self, mining_client):
        request_json = {}
        response = mining_client.post('/database', json=request_json)
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": "The request identifiers is missing."}

        request_json = "text"
        response = mining_client.post('/database', data=request_json)
        assert response.json == {"error": "The request has to be a JSON object."}

        identifiers = [('w8579f54', 4)]
        request_json = {"identifiers": identifiers}
        response = mining_client.post('/database', json=request_json)
        assert response.headers['Content-Type'] == 'text/csv'
        assert response.data.decode('utf-8').split('\n')[0] == 'entity,entity_type,property,' \
                                                               'property_value,property_type,' \
                                                               'property_value_type,' \
                                                               'ontology_source,paper_id,' \
                                                               'start_char,end_char'

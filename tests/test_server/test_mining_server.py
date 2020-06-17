"""Test for the mining server."""
from flask import Flask
import pytest
from unittest.mock import Mock, MagicMock

from bbsearch.mining.relation import ChemProt
from bbsearch.server.mining_server import MiningServer


@pytest.fixture
def mining_client(monkeypatch):
    """Fixture to create a client for mining_server."""

    chemprot_inst = MagicMock(spec=ChemProt)

    chemprot_class = Mock()
    chemprot_class.return_value = chemprot_inst

    chemprot_inst.classes.return_value = ['test1', 'test2']
    chemprot_inst.symbols.return_value = {'GGP': ('[[ ', ' ]]'), 'CHEBI': ('<< ', ' >>')}
    chemprot_inst.predict.return_value = 'test1'
    chemprot_inst.predict_probs.return_value = ''

    monkeypatch.setattr('bbsearch.server.mining_server.ChemProt', chemprot_class)

    app = Flask("BBSearch Test Mining Server")

    mining_server = MiningServer(app=app,
                                 models_path='')
    mining_server.app.config['TESTING'] = True
    with mining_server.app.test_client() as client:
        yield client


class TestMiningServer:

    def test_mining_server_help(self, mining_client):
        response = mining_client.post('/help')
        assert response.json['name'] == 'MiningServer'

    def test_mining_server_pipeline(self, mining_client):
        request_json = {
            "text": 'hello',
            "ee_models": 'en_ner_craft_md',
            "re_models": 'chemprot',
            "article_id": 5,
            "return_prob": False}
        response = mining_client.post('/', json=request_json)
        assert response.headers['Content-Type'] == 'text/csv'
        assert response.data.decode('utf-8') == 'entity,entity_type,property,property_value,property_type,' \
                                                'property_value_type,' \
                                                'ontology_source,paper_id,start_char,end_char,confidence_score\n'
        request_json = {"article_id": 5,
                        "ee_models": 'en_ner_craft_md',
                        "re_models": '',
                        "return_prob": False}
        response = mining_client.post('/', json=request_json)
        assert list(response.json.keys()) == ["error"]
        assert response.json == {"error": "The request text is missing."}

        request_json = "text"
        response = mining_client.post('/', data=request_json)
        assert response.json == {"error": "The request has to be a JSON object."}

        request_json = {"text": 'hello',
                        "article_id": 5,
                        "ee_models": '',
                        "return_prob": False}
        response = mining_client.post('/', json=request_json)
        assert list(response.json.keys()) == ["error"]

        request_json = {"text": 'hello',
                        "article_id": 5,
                        "ee_models": 'wrong_entity_model',
                        "return_prob": False}
        response = mining_client.post('/', json=request_json)
        assert list(response.json.keys()) == ["error"]

        request_json = {"text": 'hello',
                        "article_id": 5,
                        "ee_models": 'en_ner_craft_md',
                        "re_models": 'wrong_relation_model',
                        "return_prob": False}
        response = mining_client.post('/', json=request_json)
        assert list(response.json.keys()) == ["error"]

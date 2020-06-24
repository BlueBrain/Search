from flask import Flask
import pytest
from unittest.mock import Mock

import numpy as np

from bbsearch.server.search_server import SearchServer


@pytest.fixture
def search_client(monkeypatch, embeddings_path, fake_db_cnxn, test_parameters):
    """Fixture to create a client for mining_server."""

    bsv_model_inst = Mock()
    bsv_model_class = Mock()
    bsv_model_class.return_value = bsv_model_inst
    bsv_model_inst.preprocess.return_value = 'hello'
    bsv_model_inst.embed.return_value = np.ones((test_parameters['embedding_size'],))

    sbiobert_model_inst = Mock()
    sbiobert_model_class = Mock()
    sbiobert_model_class.return_value = sbiobert_model_inst
    sbiobert_model_inst.preprocess.return_value = 'hello'
    sbiobert_model_inst.embed.return_value = np.ones((test_parameters['embedding_size'],))

    monkeypatch.setattr('bbsearch.server.search_server.BSV', bsv_model_class)
    monkeypatch.setattr('bbsearch.server.search_server.SBioBERT', sbiobert_model_class)

    database_path = fake_db_cnxn.execute("""PRAGMA database_list""").fetchall()[0][2]

    app = Flask("BBSearch Test Search Server")

    search_server = SearchServer(app=app,
                                 trained_models_path='',
                                 embeddings_path=embeddings_path,
                                 database_path=database_path)
    search_server.app.config['TESTING'] = True
    with search_server.app.test_client() as client:
        yield client


class TestSearchServer:

    def test_search_server(self, search_client):

        response = search_client.post('/help')
        assert response.status_code == 200
        assert response.json['name'] == 'SearchServer'

        request_json = {'which_model': 'BSV',
                        'k': 3,
                        'query_text': 'hello'}
        response = search_client.post('/', json=request_json)
        assert response.status_code == 200

        json_response = response.json
        assert len(json_response['sentence_ids']) == 3
        assert len(json_response['similarities']) == 3

        request_json = 'data is not a json'
        response = search_client.post('/', data=request_json)
        assert response.status_code == 200

        json_response = response.json
        assert json_response['sentence_ids'] is None
        assert json_response['similarities'] is None

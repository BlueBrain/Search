import pytest

from bluesearch.k8s.connect import connect
from bluesearch.k8s.create_indices import add_index, remove_index

ES_URL = "http://localhost:9200"
ES_PASS = ""


@pytest.fixture()
def get_es_client(monkeypatch):
    monkeypatch.setenv("ES_URL", ES_URL)
    monkeypatch.setenv("ES_PASS", ES_PASS)

    try:
        client = connect()
    except RuntimeError:
        client = None

    yield client


def test_create_and_remove_index(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"

    add_index(client, index)
    remove_index(client, index)

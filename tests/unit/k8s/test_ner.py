import pytest

from bluesearch.k8s.create_indices import add_index
from bluesearch.k8s.ner import run, run_ner_model_remote, retrieve_csv

def test_create_and_remove_index(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"

    add_index(client, index)


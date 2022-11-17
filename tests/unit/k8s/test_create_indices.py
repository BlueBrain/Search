import pytest

from bluesearch.k8s.create_indices import add_index, remove_index


def test_create_and_remove_index(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"

    add_index(client, index)
    remove_index(client, index)

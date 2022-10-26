import pytest

from bluesearch.k8s.create_indices import (
    SETTINGS,
    MAPPINGS_ARTICLES,
    add_index,
    remove_index,
    update_index_mapping,
)


def test_create_and_remove_index(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"

    add_index(client, index)
    remove_index(client, index)


def test_update_index_mapping(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"

    add_index(client, index, settings=SETTINGS, mappings=MAPPINGS_ARTICLES)

    index_settings = client.indices.get_settings(index=index)
    assert index_settings[index]["settings"]["index"]["number_of_replicas"] == str(SETTINGS["number_of_replicas"])
    assert client.indices.get_mapping(index=index)[index]["mappings"] == MAPPINGS_ARTICLES

    fake_settings = {"number_of_replicas": 2}
    fake_properties = {"x": {"type": "text"}}
    update_index_mapping(
        client,
        index,
        settings=fake_settings,
        properties=fake_properties,
    )

    index_settings = client.indices.get_settings(index=index)
    assert index_settings[index]["settings"]["index"]["number_of_replicas"] == "2"

    NEW_MAPPINGS_ARTICLES = MAPPINGS_ARTICLES.copy()
    NEW_MAPPINGS_ARTICLES["properties"]["x"] = {"type": "text"}
    assert client.indices.get_mapping(index=index)[index]["mappings"] == NEW_MAPPINGS_ARTICLES

    remove_index(client, index)

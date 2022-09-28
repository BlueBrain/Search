import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_add_embeddings(get_es_client):
    from bluesearch.k8s.create_indices import (
        MAPPINGS_PARAGRAPHS,
        SETTINGS,
        add_index,
        remove_index,
    )
    from bluesearch.k8s.embeddings import embed_locally

    client = get_es_client
    if client is None:
        pytest.skip("Elastic search is not available")

    add_index(client, "test_paragraphs", SETTINGS, MAPPINGS_PARAGRAPHS)

    docs = {
        "1": {"text": "some test text"},
        "2": {"text": "some other test text"},
        "3": {"text": "some final test text"},
    }

    for doc_id, doc in docs.items():
        client.create(index="test_paragraphs", id=doc_id, document=doc)
    client.indices.refresh(index="test_paragraphs")

    query = {"bool": {"must_not": {"exists": {"field": "embedding"}}}}
    paragraph_count = client.count(index="test_paragraphs", query=query)
    assert paragraph_count["count"] == 3

    embed_locally(
        client,
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        index="test_paragraphs",
    )
    client.indices.refresh(index="test_paragraphs")

    query = {"bool": {"must_not": {"exists": {"field": "embedding"}}}}
    paragraph_count = client.count(index="test_paragraphs", query=query)["count"]
    assert paragraph_count == 0

    remove_index(client, "test_paragraphs")
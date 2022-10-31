import pytest
import responses

from bluesearch.k8s.create_indices import add_index
from bluesearch.k8s.ner import run_ner_model_remote


@responses.activate
def test_run_ner_model_remote(get_es_client):
    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"
    add_index(client, index)

    url = "fake_url"
    expected_url = "http://" + url + "/predict"
    hit = {"_source": {"text": "There is a cat and a mouse in the house."}}
    expected_response = [
        {
            "entity_group": "ORGANISM",
            "score": 0.9439833760261536,
            "word": "cat",
            "start": 11,
            "end": 14,
        },
        {
            "entity_group": "ORGANISM",
            "score": 0.9975798726081848,
            "word": "mouse",
            "start": 21,
            "end": 26,
        },
    ]

    responses.add(
        responses.POST,
        expected_url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        json=expected_response,
    )

    out = run_ner_model_remote(hit, url, ner_method="ml")
    assert isinstance(out, list)
    assert len(out) == 2

    assert out[0]["source"] == "ml"
    assert out[0]["score"] == 0.9439833760261536
    assert out[1]["score"] == 0.9975798726081848
    assert out[0]["entity_type"] == "ORGANISM"
    assert out[0]["entity"] == "cat"
    assert out[0]["start"] == 11
    assert out[0]["end"] == 14

    out = run_ner_model_remote(hit, url, ner_method="ruler")
    assert isinstance(out, list)
    assert out[0]["score"] == 0
    assert out[1]["score"] == 0

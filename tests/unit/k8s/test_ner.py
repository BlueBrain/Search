import pytest
import responses

from bluesearch.k8s.create_indices import MAPPINGS_PARAGRAPHS, add_index, remove_index
from bluesearch.k8s.ner import run, run_ner_model_remote


@pytest.fixture()
def model_response():
    url = "fake_url"
    expected_url = "http://" + url + "/predict"
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

    responses.add(
        responses.POST,
        expected_url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        json=expected_response,
    )


@responses.activate
def test_run_ner_model_remote(model_response):
    url = "fake_url"
    hit = {"_source": {"text": "There is a cat and a mouse in the house."}}

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


@responses.activate
def test_run(monkeypatch, get_es_client, model_response):

    BENTOML_NER_ML_URL = "fake_url"
    BENTOML_NER_RULER_URL = "fake_url"

    monkeypatch.setenv("BENTOML_NER_ML_URL", BENTOML_NER_ML_URL)
    monkeypatch.setenv("BENTOML_NER_RULER_URL", BENTOML_NER_RULER_URL)

    client = get_es_client

    if client is None:
        pytest.skip("Elastic search is not available")

    index = "test_index"
    add_index(client, index, mappings=MAPPINGS_PARAGRAPHS)

    fake_data = [
        {
            "article_id": "1",
            "paragraph_id": "1",
            "text": "There is a cat and a mouse in the house.",
        },
        {
            "article_id": "1",
            "paragraph_id": "2",
            "text": "There is a cat and a mouse in the house.",
        },
        {
            "article_id": "2",
            "paragraph_id": "3",
            "text": "There is a cat and a mouse in the house.",
        },
    ]

    for fd in fake_data:
        client.index(index=index, document=fd, id=fd["paragraph_id"])

    run(client, "v1", index=index, run_async=False)
    client.indices.refresh(index=index)

    # check that the results are in the database
    query = {"bool": {"must": {"term": {"ner_ml_version": "v1"}}}}
    paragraph_count = client.count(index=index, query=query)["count"]
    assert paragraph_count == 3

    # check that the results are correct
    for fd in fake_data:
        res = client.get(index=index, id=fd["paragraph_id"])
        assert res["_source"]["ner_ml_json_v2"][0]["entity"] == "cat"
        assert res["_source"]["ner_ml_json_v2"][0]["score"] == 0.9439833760261536
        assert res["_source"]["ner_ml_json_v2"][0]["entity_type"] == "ORGANISM"
        assert res["_source"]["ner_ml_json_v2"][0]["start"] == 11
        assert res["_source"]["ner_ml_json_v2"][0]["end"] == 14
        assert res["_source"]["ner_ml_json_v2"][0]["source"] == "ml"

        assert res["_source"]["ner_ml_json_v2"][1]["entity"] == "mouse"
        assert res["_source"]["ner_ml_json_v2"][1]["score"] == 0.9975798726081848
        assert res["_source"]["ner_ml_json_v2"][1]["entity_type"] == "ORGANISM"
        assert res["_source"]["ner_ml_json_v2"][1]["start"] == 21
        assert res["_source"]["ner_ml_json_v2"][1]["end"] == 26
        assert res["_source"]["ner_ml_json_v2"][1]["source"] == "ml"

        assert res["_source"]["ner_ml_version"] == "v1"

    # check that all paragraphs have been updated
    query = {"bool": {"must_not": {"term": {"ner_ml_version": "v1"}}}}
    paragraph_count = client.count(index=index, query=query)["count"]
    assert paragraph_count == 0

    remove_index(client, index)

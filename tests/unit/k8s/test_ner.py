import pytest
import responses

from bluesearch.k8s.create_indices import MAPPINGS_PARAGRAPHS, add_index, remove_index
from bluesearch.k8s.ner import handle_conflicts, run, run_ner_model_remote


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

    client.indices.refresh(index=index)
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


@pytest.mark.parametrize(
    ("raw_ents", "cleaned_ents"),
    [
        pytest.param([], [], id="empty list"),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                },
            ],
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                },
            ],
            id="one element",
        ),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 10,
                    "end": 15,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 10,
                    "end": 15,
                    "word": "word",
                    "source": "ML",
                },
            ],
            id="no overlap",
        ),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "ML",
                },
            ],
            id="perfect overlap",
        ),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 5,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 2,
                    "end": 20,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 2,
                    "end": 20,
                    "word": "word",
                    "source": "ML",
                },
            ],
            id="overlap - ML longer",
        ),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 50,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 25,
                    "end": 60,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 1,
                    "end": 50,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            id="overlap - RULES longer",
        ),
        pytest.param(
            [
                {
                    "start": 1,
                    "end": 50,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 25,
                    "end": 40,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 1,
                    "end": 50,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            id="overlap - ML subset of RULES",
        ),
        pytest.param(
            [
                {
                    "start": 4,
                    "end": 24,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 2,
                    "end": 40,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 2,
                    "end": 40,
                    "word": "word",
                    "source": "ML",
                },
            ],
            id="overlap - RULES subset of ML",
        ),
        pytest.param(
            [
                {
                    "start": 10,
                    "end": 30,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 20,
                    "end": 40,
                    "word": "word",
                    "source": "ML",
                },
            ],
            [
                {
                    "start": 20,
                    "end": 40,
                    "word": "word",
                    "source": "ML",
                },
            ],
            id="overlap - same length",
        ),
        pytest.param(
            [
                {
                    "start": 10,
                    "end": 30,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 15,
                    "end": 20,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 23,
                    "end": 34,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 31,
                    "end": 33,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            [
                {
                    "start": 10,
                    "end": 30,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 31,
                    "end": 33,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            id="more entries - 1",
        ),
        pytest.param(
            [
                {
                    "start": 10,
                    "end": 30,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 20,
                    "end": 50,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 35,
                    "end": 100,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            [
                {
                    "start": 10,
                    "end": 30,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 35,
                    "end": 100,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            id="more entries - 2",
        ),
        pytest.param(
            [
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 35,
                    "end": 100,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            [
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 35,
                    "end": 100,
                    "word": "word",
                    "source": "RULES",
                },
            ],
            id="entries with only 2 chars",
        ),
        pytest.param(
            [
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "RULES",
                },
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 12,
                    "end": 15,
                    "word": " word",
                    "source": "RULES",
                },
            ],
            [
                {
                    "start": 10,
                    "end": 12,
                    "word": "word",
                    "source": "ML",
                },
                {
                    "start": 12,
                    "end": 15,
                    "word": " word",
                    "source": "RULES",
                },
            ],
            id="overlap with whitespace",
        ),
    ],
)
def test_handle_conflicts(raw_ents, cleaned_ents):
    """Test handle_conflicts function."""
    assert cleaned_ents == handle_conflicts(raw_ents)

import pytest
import responses

from bluesearch.k8s.relation_extraction import run_re_model_remote


@pytest.fixture()
def model_response():
    url = "fake_url"
    expected_url = "http://" + url + "/predict"
    expected_response = [
        {"label": "IS IN", "score": 0.999029278755188},
        {"label": "IS IN", "score": 0.9987483024597168},
    ]

    responses.add(
        responses.POST,
        expected_url,
        headers={"accept": "application/json", "Content-Type": "application/json"},
        json=expected_response,
    )


@responses.activate
def test_run_re_model_remote(model_response):
    url = "fake_url"
    hit = {
        "_source": {
            "text": "There is a mouse and a cat in the house.",
            "ner_ml_json_v2": [
                {
                    "entity_type": "CELL_TYPE",
                    "score": 0.9943312406539917,
                    "entity": "cell",
                    "start": 11,
                    "end": 15,
                    "source": "ml",
                },
                {
                    "entity_type": "CELL_COMPARTMENT",
                    "score": 0.999178946018219,
                    "entity": "axon",
                    "start": 23,
                    "end": 27,
                    "source": "ml",
                },
                {
                    "entity_type": "BRAIN_REGION",
                    "score": 0.999150276184082,
                    "entity": "hippocampus",
                    "start": 35,
                    "end": 46,
                    "source": "ml",
                },
            ],
            "ner_ruler_json_v2": [
                {
                    "entity_type": "CELL_TYPE",
                    "entity": "cell",
                    "start": 11,
                    "end": 15,
                    "source": "ruler",
                },
                {
                    "entity_type": "CELL_COMPARTMENT",
                    "entity": "axon",
                    "start": 23,
                    "end": 27,
                    "source": "ruler",
                },
                {
                    "entity_type": "BRAIN_REGION",
                    "entity": "hippocampus",
                    "start": 35,
                    "end": 46,
                    "source": "ruler",
                },
            ],
        },
    }

    out = run_re_model_remote(hit, url)
    assert isinstance(out, list)
    assert len(out) == 2

    assert out[0] == {
        "label": "IS IN",
        "score": 0.999029278755188,
        "subject_entity_type": "CELL_TYPE",
        "subject_entity": "cell",
        "subject_start": 11,
        "subject_end": 15,
        "subject_source": "ml",
        "object_entity_type": "BRAIN_REGION",
        "object_entity": "hippocampus",
        "object_start": 35,
        "object_end": 46,
        "object_source": "ml",
        "source": "ml",
    }

    assert out[1] == {
        "label": "IS IN",
        "score": 0.9987483024597168,
        "subject_entity_type": "CELL_COMPARTMENT",
        "subject_entity": "axon",
        "subject_start": 23,
        "subject_end": 27,
        "subject_source": "ml",
        "object_entity_type": "CELL_TYPE",
        "object_entity": "cell",
        "object_start": 11,
        "object_end": 15,
        "object_source": "ml",
        "source": "ml",
    }

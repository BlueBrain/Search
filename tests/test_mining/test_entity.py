"""Collections of tests covering the `entity.py` module."""
import pathlib

import pandas as pd

from bbsearch.mining import (
    dump_jsonl,
    global2model_patterns,
    load_jsonl,
    remap_entity_type,
)


def test_load_jsonl(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "file.jsonl"

    li = [{"a": 1, "b": "cc"}, {"k": 23}]
    dump_jsonl(li, path)
    lo = load_jsonl(path)

    assert li == lo


def test_entity_type():
    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "DISEASE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]

    etype_mapping = {"CHEMICAL": "CHEBI"}

    adjusted_patterns = remap_entity_type(patterns, etype_mapping)
    adjusted_patterns_true = [
        {"label": "NaE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "NaE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEBI", "pattern": [{"LOWER": "glucose"}]},
    ]

    assert patterns is not adjusted_patterns
    assert adjusted_patterns == adjusted_patterns_true


def test_global2model_patterns():
    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "DISEASE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]

    ee_models_library = pd.DataFrame(
        [
            ["CHEMICAL", "model_1", "CHEBI"],
            ["ORGANISM", "model_2", "ORG"],
            ["DISEASE", "model_3", "DISEASE"],
        ],
        columns=["entity_type", "model", "entity_type_name"],
    )

    res = global2model_patterns(patterns, ee_models_library)

    assert len(res) == 3
    assert set(res.keys()) == set(ee_models_library["model"].unique())

    model_1_patterns = [
        {"label": "NaE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "NaE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEBI", "pattern": [{"LOWER": "glucose"}]},
    ]
    model_2_patterns = [
        {"label": "NaE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "NaE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "NaE", "pattern": [{"LOWER": "glucose"}]},
    ]
    model_3_patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "DISEASE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "NaE", "pattern": [{"LOWER": "glucose"}]},
    ]

    assert res["model_1"] == model_1_patterns
    assert res["model_2"] == model_2_patterns
    assert res["model_3"] == model_3_patterns

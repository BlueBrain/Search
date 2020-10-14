"""Collections of tests covering the `entity.py` module."""
import pandas as pd
import pytest
import spacy

from bbsearch.mining import (
    check_patterns_agree,
    global2model_patterns,
    remap_entity_type,
)


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


def test_check_patterns_agree():
    model = spacy.blank("en")

    # No entity rulers
    with pytest.raises(ValueError):
        check_patterns_agree(model, [])

    er_1 = spacy.pipeline.EntityRuler(model)
    er_2 = spacy.pipeline.EntityRuler(model)

    # Single entity ruler
    model.add_pipe(er_1, first=True, name="er_1")

    assert check_patterns_agree(model, [])

    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "DISEASE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]
    er_1.add_patterns(patterns)

    assert check_patterns_agree(model, patterns)
    assert not check_patterns_agree(
        model, patterns[::-1]
    )  # unfortunately the order matters

    # Two entity rules
    model.add_pipe(er_2, first=True, name="er_2")

    with pytest.raises(ValueError):
        check_patterns_agree(model, [])

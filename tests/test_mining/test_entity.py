"""Collections of tests covering the `entity.py` module."""
import pathlib

import pandas as pd
import pytest
import spacy

from bbsearch.mining import (
    PatternCreator,
    check_patterns_agree,
    global2model_patterns,
    remap_entity_type,
)


class TestPatternCreator:
    def test_equality(self):
        pc1 = PatternCreator()
        pc2 = PatternCreator()
        pc2.add("ETYPE", "hello")

        assert pc1 == pc1
        assert pc1 != "wrong type"
        assert pc1 != pc2  # different columns

    def test_errors(self):
        pc = PatternCreator()

        with pytest.raises(TypeError):
            pc.add("ETYPE", 234324)  # invalid type

        pc.add("ETYPE", "hello")

        with pytest.raises(ValueError):
            pc.add("ETYPE", "hello", check_exists=True)  # duplicate

        with pytest.raises(ValueError):
            pc.add("etype", [{"a": 1, "b": 2}])  # wrong contents

    def test_overall(self, tmpdir):
        tmpdir_p = pathlib.Path(str(tmpdir)) / "patterns.json"

        pc = PatternCreator()

        assert len(pc.to_df()) == 0

        pc.add("NEW_ENTITY_TYPE", "cake")

        assert len(pc.to_df()) == 1
        assert set(pc.to_df().columns) == {"label",
                                           "attribute_0",
                                           "value_0",
                                           "value_type_0",
                                           "op_0"}

        pc.add("COOL_ENTITY_TYPE", {"LEMMA": "pancake", "OP": "*"})

        assert len(pc.to_df()) == 2

        pc.add("SOME_ENTITY_TYPE", [{"TEXT": "good"}, {"TEXT": "pizza"}])

        assert len(pc.to_df()) == 3
        assert set(pc.to_df().columns) == {"label",
                                           "attribute_0",
                                           "value_0",
                                           "value_type_0",
                                           "op_0",
                                           "attribute_1",
                                           "value_1",
                                           "value_type_1",
                                           "op_1",
                                           }

        pc.save(tmpdir_p)
        pc_loaded = PatternCreator.load(tmpdir_p)
        pc_manual = PatternCreator(storage=pc.to_df())

        assert pc == pc_loaded == pc_manual

    def test_test(self):
        pc = PatternCreator()

        pc.add("new_entity_type", "tall")

        text = "I saw a tall building."
        doc = pc.test(text)
        assert len(doc.ents) == 1
        assert list(doc.ents)[0].label_ == "new_entity_type"

        pc.drop(0)

        doc2 = pc.test(text)

        assert len(doc2.ents) == 0


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

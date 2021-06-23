"""Collections of tests covering the `entity.py` module."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pathlib

import numpy as np
import pandas as pd
import pytest
import spacy

from bluesearch.mining import (
    PatternCreator,
    check_patterns_agree,
    global2model_patterns,
)


class TestPatternCreator:
    def test_equality(self):
        pc1 = PatternCreator()
        pc2 = PatternCreator()
        pc2.add("ETYPE", "hello")

        assert pc1 == pc1
        assert pc1 != "wrong type"
        assert pc1 != pc2  # different columns

    def test_drop(self):
        pc = PatternCreator()

        pc.add("ET1", "hello")
        pc.add("ET1", "there")
        pc.add("ET2", "world")
        pc.add("ET4", "dog")

        assert pc.to_df().index.to_list() == [0, 1, 2, 3]

        pc.drop([1, 2])

        assert pc.to_df().index.to_list() == [0, 1]

    def test_to_df(self):
        pc = PatternCreator()

        pc.add("ET1", "hello")
        pc.add("ET1", "there")

        df_1 = pc.to_df()
        df_2 = pc.to_df()

        df_2.loc[0, "label"] = "REPLACED_LABEL"

        df_3 = pc.to_df()

        assert not df_1.equals(df_2)
        assert df_1.equals(df_3)

    def test_to_list(self):
        pc = PatternCreator()

        pc.add("ET1", "hello")
        pc.add("ET2", {"TEXT": "there"})
        pc.add("ET3", [{"TEXT": {"IN": ["world", "cake"]}}])
        pc.add("ET4", [{"TEXT": {"IN": ["aa", "bbb"]}}, {"TEXT": {"REGEX": "^s"}}])

        res = pc.to_list()

        assert len(res) == 4

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
        assert set(pc.to_df().columns) == {
            "label",
            "attribute_0",
            "value_0",
            "value_type_0",
            "op_0",
        }

        pc.add("COOL_ENTITY_TYPE", {"LEMMA": "pancake", "OP": "*"})

        assert len(pc.to_df()) == 2

        pc.add("SOME_ENTITY_TYPE", [{"TEXT": "good"}, {"TEXT": "pizza"}])

        assert len(pc.to_df()) == 3
        assert set(pc.to_df().columns) == {
            "label",
            "attribute_0",
            "value_0",
            "value_type_0",
            "op_0",
            "attribute_1",
            "value_1",
            "value_type_1",
            "op_1",
        }

        pc.to_jsonl(tmpdir_p)
        pc_loaded = PatternCreator.from_jsonl(tmpdir_p)
        pc_manual = PatternCreator(storage=pc.to_df())

        assert pc == pc_loaded == pc_manual

    def test_call(self):
        pc = PatternCreator()

        pc.add("new_entity_type", "tall")

        text = "I saw a tall building."
        doc = pc(text)
        assert len(doc.ents) == 1
        assert list(doc.ents)[0].label_ == "new_entity_type"

        pc.drop(0)

        doc2 = pc(text)

        assert len(doc2.ents) == 0

    def test_raw2row(self):
        # pattern not a list
        with pytest.raises(TypeError):
            PatternCreator.raw2row({"label": "ET1", "pattern": {"LOWER": "TEXT"}})

        # label not a str
        with pytest.raises(TypeError):
            PatternCreator.raw2row({"label": 232, "pattern": [{"LOWER": "TEXT"}]})

        # element not dictionary
        with pytest.raises(TypeError):
            PatternCreator.raw2row({"label": "etype", "pattern": [11]})

    def test_row2raw(self):
        # unsupported value_type - eval fails
        with pytest.raises(NameError):
            PatternCreator.row2raw(
                pd.Series(
                    {
                        "label": "et1",
                        "attribute_0": "TEXT",
                        "value_0": "aaa",
                        "value_type_0": "wrong_type",
                        "op_0": "",
                    }
                )
            )

        # already the first token is invalid
        with pytest.raises(ValueError):
            PatternCreator.row2raw(
                pd.Series(
                    {
                        "label": "et1",
                        "attribute_0": np.nan,
                        "value_0": "aaa",
                        "value_type_0": "wrong_type",
                        "op_0": "",
                    }
                )
            )

        res = PatternCreator.row2raw(
            pd.Series(
                {
                    "label": "et1",
                    "attribute_0": "TEXT",
                    "value_0": "aaa",
                    "value_type_0": "str",
                    "op_0": "",
                    "attribute_1": np.nan,
                    "value_1": "bbb",
                    "value_type_1": "int",
                    "op_1": "!",
                }
            )
        )

        assert res == {"label": "et1", "pattern": [{"TEXT": "aaa"}]}

    @pytest.mark.parametrize(
        "raw",
        [
            {"label": "ET1", "pattern": [{"LOWER": "something"}]},
            {"label": "ET2", "pattern": [{"REGEX": "^S"}]},
            {"label": "ET3", "pattern": [{"LEMMA": "bb", "OP": "!"}]},
            {"label": "ET4", "pattern": [{"OP": "+", "LOWER": "fdsaf"}]},
            {"label": "ET5", "pattern": [{"ORTH": "fdsaf"}, {"LEMMA": "aaa"}]},
            {
                "label": "ET6",
                "pattern": [
                    {"OP": "+", "LOWER": "aaa"},
                    {"OP": "!", "LEMMA": "bb"},
                    {"OP": "?", "ORTH": "cc"},
                ],
            },
            {"label": "ET7", "pattern": [{"LENGTH": 5}]},
            {
                "label": "ET8",
                "pattern": [{"TEXT": {"IN": ["aa", "bbb"]}}, {"TEXT": {"REGEX": "^s"}}],
            },
        ],
    )
    def test_raw2row2raw(self, raw):
        assert raw == PatternCreator.row2raw(PatternCreator.raw2row(raw))


def test_entity_type():
    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "DISEASE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]

    adjusted_patterns = global2model_patterns(patterns, "CHEMICAL")
    adjusted_patterns_true = [
        {"label": "NaE", "pattern": [{"LOWER": "covid-19"}]},
        {
            "label": "NaE",
            "pattern": [{"LOWER": "covid"}, {"TEXT": "-"}, {"TEXT": "19"}],
        },
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]

    assert patterns is not adjusted_patterns
    assert adjusted_patterns == adjusted_patterns_true


def test_check_patterns_agree():
    model = spacy.blank("en")

    # No entity rulers
    with pytest.raises(ValueError):
        check_patterns_agree(model, [])

    # Single entity ruler
    er_1 = model.add_pipe("entity_ruler", first=True, name="er_1")

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
    model.add_pipe("entity_ruler", first=True, name="er_2")

    with pytest.raises(ValueError):
        check_patterns_agree(model, [])

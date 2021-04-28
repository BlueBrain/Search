"""Tests covering functionalities for the evaluation of mining tools."""

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

import json
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import spacy

from bluesearch.mining import annotations2df, spacy2df
from bluesearch.mining.eval import (
    _check_consistent_iob,
    idx2text,
    iob2idx,
    ner_confusion_matrix,
    ner_errors,
    ner_report,
    remove_punctuation,
    unique_etypes,
)


class TestAnnotations2df:
    @pytest.mark.parametrize("answer", ["accept", "ignore"])
    def test_overall(self, answer, tmpdir):
        tmp_dir = pathlib.Path(str(tmpdir))
        tmp_file = tmp_dir / "annot.jsonl"

        prodigy_content = {
            "answer": answer,
            "meta": {"pattern": "", "source": "amazing source"},
            "spans": [
                {
                    "label": "PERSON",
                    "start": 0,
                    "end": 14,
                    "token_start": 0,
                    "token_end": 1,
                },
                {
                    "label": "GPE",
                    "start": 32,
                    "end": 38,
                    "token_start": 6,
                    "token_end": 6,
                },
                {
                    "label": "DATE",
                    "start": 39,
                    "end": 48,
                    "token_start": 7,
                    "token_end": 7,
                },
            ],
            "tokens": [
                {"text": "Britney", "start": 0, "end": 7, "id": 0},
                {"text": "Spears", "start": 8, "end": 14, "id": 1},
                {"text": "had", "start": 15, "end": 18, "id": 2},
                {"text": "a", "start": 19, "end": 20, "id": 3},
                {"text": "concert", "start": 21, "end": 28, "id": 4},
                {"text": "in", "start": 29, "end": 31, "id": 5},
                {"text": "Brazil", "start": 32, "end": 38, "id": 6},
                {"text": "yesterday", "start": 39, "end": 48, "id": 7},
                {"text": ".", "start": 48, "end": 49, "id": 8},
            ],
        }

        # write example twice, but the second one w/o annotations
        with tmp_file.open("w") as f:
            f.write(json.dumps(prodigy_content) + "\n")
            del prodigy_content["spans"]
            f.write(json.dumps(prodigy_content) + "\n")

        df = annotations2df(tmp_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2 * (
            len(prodigy_content["tokens"]) if answer == "accept" else 0
        )

        if answer == "accept":
            assert {"source", "class", "start_char", "end_char", "id", "text"} == set(
                df.columns
            )
            # for the first example, annotations are in IOB mode
            assert set(df["class"][: len(df) // 2]) == {
                "O",
                "B-PERSON",
                "I-PERSON",
                "B-GPE",
                "B-DATE",
            }

            # for the second example, no annotations
            assert set(df["class"][len(df) // 2 :]) == {"O"}

        # test that it works for more than one file
        df = annotations2df([tmp_file] * 5)
        assert isinstance(df, pd.DataFrame)

        assert len(df) == 5 * 2 * (
            len(prodigy_content["tokens"]) if answer == "accept" else 0
        )


class TestSpacy2df:
    def test_overall(self, model_entities):
        ground_truth_tokenization = [
            "Britney",
            "Spears",
            "had",
            "a",
            "concert",
            "in",
            "Brazil",
            "yesterday",
            ".",
        ]
        classes = ["B-PERSON", "I-PERSON", "O", "O", "O", "O", "B-GPE", "B-DATE", "O"]

        df = spacy2df(model_entities, ground_truth_tokenization)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(ground_truth_tokenization)
        assert df["class"].to_list() == classes
        assert "class" in df.columns
        assert "text" in df.columns

    @pytest.mark.parametrize(
        "overwrite_ents", [True, False], ids=["overwrite", "dont_overwrite"]
    )
    @pytest.mark.parametrize("excluded_entity_type", [None, "ET1", "ET2"])
    def test_exclusion(self, overwrite_ents, excluded_entity_type, model_entities):
        ground_truth_tokenization = [
            "Britney",
            "Spears",
            "had",
            "a",
            "concert",
            "in",
            "Brazil",
            "yesterday",
            ".",
        ]

        # 1st level = overwrite_ents, 2nd level = excluded_entity_type
        classes_true = {
            True: {
                None: [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "B-ET2",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
                "ET1": [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "B-ET2",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
                "ET2": [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
            },
            False: {
                None: [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "B-ET1",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
                "ET1": [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
                "ET2": [
                    "B-PERSON",
                    "I-PERSON",
                    "O",
                    "O",
                    "B-ET1",
                    "O",
                    "B-GPE",
                    "B-DATE",
                    "O",
                ],
            },
        }
        model = spacy.blank("en")

        model.add_pipe("ner", first=True, source=model_entities)
        er1 = model.add_pipe("entity_ruler", name="er_1", last=True)
        er1.add_patterns([{"label": "ET1", "pattern": "concert"}])
        er2 = model.add_pipe(
            "entity_ruler",
            name="er_2",
            last=True,
            config={"overwrite_ents": overwrite_ents},
        )
        er2.add_patterns([{"label": "ET2", "pattern": "concert"}])

        df = spacy2df(
            model, ground_truth_tokenization, excluded_entity_type=excluded_entity_type
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(ground_truth_tokenization)
        assert (
            df["class"].to_list() == classes_true[overwrite_ents][excluded_entity_type]
        )
        assert "class" in df.columns
        assert "text" in df.columns


@pytest.mark.parametrize(
    "dataset, annotator, etypes, counts",
    [
        (
            "bio",
            "annotator_1",
            ["CONDITION", "DISEASE", "ORGANISM", "PATHWAY"],
            {"entity": [2, 4, 6, 2], "token": [3, 9, 9, 4]},
        ),
        (
            "bio",
            "annotator_2",
            ["CONDITION", "DISEASE", "PATHWAY", "TAXON"],
            {"entity": [1, 6, 1, 8], "token": [1, 11, 2, 11]},
        ),
        (
            "sample",
            "annotator_1",
            ["a", "b", "d"],
            {"entity": [4, 3, 1], "token": [5, 4, 1]},
        ),
        ("sample", "annotator_2", ["b", "c"], {"entity": [2, 6], "token": [4, 8]}),
    ],
)
def test_unique_etypes(ner_annotations, dataset, annotator, etypes, counts):
    for mode in ("entity", "token"):
        assert (
            unique_etypes(
                ner_annotations[dataset][annotator], return_counts=False, mode=mode
            )
            == etypes
        )
        assert unique_etypes(
            ner_annotations[dataset][annotator], return_counts=True, mode=mode
        ) == (etypes, counts[mode])


@pytest.mark.parametrize(
    "dataset, annotator, etype, idxs",
    [
        ("bio", "annotator_1", "CONDITION", [[103, 104], [108, 108]]),
        ("bio", "annotator_1", "DISEASE", [[34, 37], [40, 40], [120, 121], [148, 149]]),
        ("bio", "annotator_2", "PATHWAY", [[135, 136]]),
        ("bio", "annotator_1", "POTATOES", None),
        ("sample", "annotator_1", "b", [[2, 2], [8, 9], [13, 13]]),
        (
            "sample",
            "annotator_2",
            "c",
            [[0, 0], [1, 2], [3, 3], [5, 5], [10, 11], [12, 12]],
        ),
    ],
)
def test_iob2idx(ner_annotations, dataset, annotator, etype, idxs):
    if idxs is not None:
        pd.testing.assert_frame_equal(
            iob2idx(ner_annotations[dataset][annotator], etype),
            pd.DataFrame(data=idxs, columns=["start", "end"]),
        )
    else:
        pd.testing.assert_frame_equal(
            iob2idx(ner_annotations[dataset][annotator], etype),
            pd.DataFrame(
                data={"start": [], "end": []}, index=pd.Int64Index([]), dtype="int64"
            ),
        )


@pytest.mark.parametrize(
    "dataset, annotator, etype, texts",
    [
        ("bio", "annotator_1", "CONDITION", ["worldwide outbreak", "hospitalization"]),
        (
            "bio",
            "annotator_1",
            "ORGANISM",
            [
                "human coronaviruses",
                "Human coronaviruses",
                "Human coronaviruses",
                "HCoVs",
                "infant",
                "infant",
            ],
        ),
        ("bio", "annotator_2", "PATHWAY", ["respiratory immunity"]),
        ("bio", "annotator_1", "POTATOES", []),
    ],
)
def test_idx2text(ner_annotations, dataset, annotator, etype, texts):
    assert (
        texts
        == idx2text(
            ner_annotations[dataset]["text"],
            iob2idx(ner_annotations[dataset][annotator], etype),
        ).tolist()
    )


@pytest.mark.parametrize(
    "dataset, mode, etypes_map, dict_tp_fn_fp",
    [
        (
            "bio",
            "entity",
            {"ORGANISM": "TAXON"},
            {
                "CONDITION": [1, 1, 0],
                "DISEASE": [3, 1, 3],
                "PATHWAY": [1, 1, 0],
                "ORGANISM": [6, 0, 2],
            },
        ),
        (
            "bio",
            "token",
            {"ORGANISM": "TAXON"},
            {
                "CONDITION": [1, 2, 0],
                "DISEASE": [8, 1, 3],
                "PATHWAY": [2, 2, 0],
                "ORGANISM": [9, 0, 2],
            },
        ),
        (
            "bio",
            "entity",
            None,
            {
                "CONDITION": [1, 1, 0],
                "DISEASE": [3, 1, 3],
                "PATHWAY": [1, 1, 0],
                "ORGANISM": [0, 6, 0],
            },
        ),
        (
            "bio",
            "token",
            None,
            {
                "CONDITION": [1, 2, 0],
                "DISEASE": [8, 1, 3],
                "PATHWAY": [2, 2, 0],
                "ORGANISM": [0, 9, 0],
            },
        ),
        (
            "sample",
            "entity",
            {"a": "c"},
            {"a": [2, 2, 4], "b": [1, 2, 1], "d": [0, 1, 0]},
        ),
        (
            "sample",
            "token",
            {"a": "c"},
            {"a": [4, 1, 4], "b": [3, 1, 1], "d": [0, 1, 0]},
        ),
        ("sample_nested", "entity", {"a": "a"}, {"a": [1, 1, 3]}),
        ("sample_nested", "token", {"a": "a"}, {"a": [4, 0, 1]}),
    ],
)
def test_ner_report(ner_annotations, dataset, mode, etypes_map, dict_tp_fn_fp):
    report_str = ner_report(
        ner_annotations[dataset]["annotator_1"],
        ner_annotations[dataset]["annotator_2"],
        mode=mode,
        etypes_map=etypes_map,
        return_dict=False,
    )
    report_dict = ner_report(
        ner_annotations[dataset]["annotator_1"],
        ner_annotations[dataset]["annotator_2"],
        mode=mode,
        etypes_map=etypes_map,
        return_dict=True,
    )

    assert isinstance(report_str, str)
    assert isinstance(report_dict, OrderedDict)

    etypes = sorted(dict_tp_fn_fp.keys())
    assert list(report_dict.keys()) == etypes
    for etype in etypes:
        assert set(report_dict[etype].keys()) == {
            "precision",
            "recall",
            "f1-score",
            "support",
        }
        tp, fn, fp = dict_tp_fn_fp[etype]
        tot_true_pos = tp + fn
        tot_pred_pos = tp + fp
        prec_ = (tp / tot_pred_pos) if tot_pred_pos > 0 else 0
        recall_ = tp / tot_true_pos
        f1_ = (2 * prec_ * recall_ / (prec_ + recall_)) if tot_pred_pos > 0 else 0
        np.testing.assert_almost_equal(prec_, report_dict[etype]["precision"])
        np.testing.assert_almost_equal(recall_, report_dict[etype]["recall"])
        np.testing.assert_almost_equal(f1_, report_dict[etype]["f1-score"])
        np.testing.assert_almost_equal(tot_true_pos, report_dict[etype]["support"])


@pytest.mark.parametrize(
    "dataset, mode, cm_vals",
    [
        (
            "bio",
            "token",
            [
                [1, 0, 0, 0, 2],
                [0, 8, 0, 0, 1],
                [0, 0, 0, 9, 0],
                [0, 0, 2, 0, 2],
                [0, 3, 0, 2, 170],
            ],
        ),
        (
            "bio",
            "entity",
            [
                [1, 0, 0, 0, 1],
                [0, 3, 0, 0, 1],
                [0, 0, 0, 6, 0],
                [0, 0, 1, 0, 1],
                [0, 3, 0, 2, 0],
            ],
        ),
        ("sample", "token", [[0, 4, 1], [3, 1, 0], [0, 1, 0], [1, 2, 1]]),
        ("sample", "entity", [[0, 2, 2], [1, 0, 2], [0, 1, 0], [1, 3, 0]]),
        ("sample_nested", "token", [[4, 0], [1, 2]]),
        ("sample_nested", "entity", [[1, 1], [3, 0]]),
    ],
)
def test_ner_confusion_matrix(ner_annotations, dataset, mode, cm_vals):
    iob_true = ner_annotations[dataset]["annotator_1"]
    iob_pred = ner_annotations[dataset]["annotator_2"]
    cm_vals = np.array(cm_vals)

    cm = ner_confusion_matrix(
        iob_true=iob_true, iob_pred=iob_pred, normalize=None, mode=mode
    )
    assert isinstance(cm, pd.DataFrame)
    assert cm.index.tolist() == (unique_etypes(iob_true) + ["None"])
    assert cm.columns.tolist() == (unique_etypes(iob_pred) + ["None"])
    np.testing.assert_almost_equal(cm.values, cm_vals)

    cm_1 = ner_confusion_matrix(
        iob_true=iob_true, iob_pred=iob_pred, normalize="true", mode=mode
    )
    cm_2 = ner_confusion_matrix(
        iob_true=iob_true, iob_pred=iob_pred, normalize="pred", mode=mode
    )

    np.testing.assert_almost_equal(
        cm_1.values, cm_vals / cm_vals.sum(axis=1, keepdims=True)
    )
    np.testing.assert_almost_equal(
        cm_2.values, cm_vals / cm_vals.sum(axis=0, keepdims=True)
    )


@pytest.mark.parametrize(
    "dataset, mode, errors_expected",
    [
        (
            "bio",
            "token",
            [
                (
                    "CONDITION",
                    {"false_neg": ["outbreak", "worldwide"], "false_pos": []},
                ),
                (
                    "DISEASE",
                    {
                        "false_neg": [","],
                        "false_pos": ["OC43", "infection", "infection"],
                    },
                ),
                ("ORGANISM", {"false_neg": [], "false_pos": ["children", "children"]}),
                ("PATHWAY", {"false_neg": ["infection", "rate"], "false_pos": []}),
            ],
        ),
        (
            "bio",
            "entity",
            [
                ("CONDITION", {"false_neg": ["worldwide outbreak"], "false_pos": []}),
                (
                    "DISEASE",
                    {
                        "false_neg": ["respiratory tract infection ,"],
                        "false_pos": [
                            "OC43 infection",
                            "infection",
                            "respiratory tract infection",
                        ],
                    },
                ),
                ("ORGANISM", {"false_neg": [], "false_pos": ["children", "children"]}),
                ("PATHWAY", {"false_neg": ["infection rate"], "false_pos": []}),
            ],
        ),
        (
            "sample_nested",
            "token",
            [("a", {"false_neg": [], "false_pos": ["disease"]})],
        ),
        (
            "sample_nested",
            "entity",
            [
                (
                    "a",
                    {
                        "false_neg": ["Sars Cov-2 infection"],
                        "false_pos": ["Cov-2 infection", "Sars", "disease"],
                    },
                )
            ],
        ),
    ],
)
def test_ner_errors(ner_annotations, dataset, mode, errors_expected):
    iob_true = ner_annotations[dataset]["annotator_1"]
    iob_pred = ner_annotations[dataset]["annotator_2"]
    tokens = ner_annotations[dataset]["text"]
    etypes_map = {"ORGANISM": "TAXON"}
    errors_out = ner_errors(
        iob_true, iob_pred, tokens, mode=mode, etypes_map=etypes_map, return_dict=True
    )
    errors_expected = OrderedDict(errors_expected)
    assert errors_out == errors_expected
    with pytest.raises(ValueError):
        ner_errors(iob_true, iob_pred[:-1], tokens)


def test_remove_punctuation(punctuation_annotations):
    df_after = remove_punctuation(punctuation_annotations["before"])
    pd.testing.assert_frame_equal(df_after, punctuation_annotations["after"])


@pytest.mark.parametrize(
    "iob_pred, raises",
    [
        (pd.Series(["O", "B-a", "B-a", "I-a", "B-c", "O"]), False),
        (
            pd.Series(["O", "B-a", "B-a", "I-a", "B-c"]),
            "target variables with inconsistent numbers of samples",
        ),
        (
            pd.Series(["O", "blah", "B-a", "I-a", "B-c", "O"]),
            "label must be one of",
        ),
        (
            pd.Series(["O", "B-a", "B-a", "I-a", "I-c", "O"]),
            "should follow one of",
        ),
        (
            pd.Series(["I-a", "B-a", "B-a", "I-a", "I-a", "O"]),
            "should follow one of",
        ),
        (
            pd.Series(["O", "B-a", "B-a", "I-a", "O", "I-a"]),
            "should follow one of",
        ),
    ],
)
def test_check_consistent_iob(iob_pred, raises):
    iob_true = pd.Series(["B-a", "O", "B-a", "I-a", "I-a", "B-a"])
    if not raises:
        _check_consistent_iob(iob_true, iob_pred)
    else:
        with pytest.raises(ValueError, match=fr".*{raises}.*"):
            _check_consistent_iob(iob_true, iob_pred)

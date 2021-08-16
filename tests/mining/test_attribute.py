"""Tests covering attribute extraction."""

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
from copy import deepcopy
from typing import Dict, Set
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import spacy
from IPython.display import HTML

from bluesearch.mining import AttributeAnnotationTab, AttributeExtractor


@pytest.fixture(scope="session")
def example_results():
    text = "Example text is between 5-6 meters long and the sun is 5°C hot today."
    measurements = [
        {
            "type": "interval",
            "quantityLeast": {
                "type": "length",
                "rawValue": "5",
                "rawUnit": {
                    "name": "meters",
                    "type": "length",
                    "system": "SI base",
                    "offsetStart": 28,
                    "offsetEnd": 34,
                },
                "parsedValue": {
                    "name": "5",
                    "numeric": 5,
                    "structure": {"type": "NUMBER", "formatted": "5"},
                    "parsed": "5",
                },
                "normalizedQuantity": 5,
                "normalizedUnit": {"name": "m", "type": "length", "system": "SI base"},
                "offsetStart": 24,
                "offsetEnd": 25,
            },
            "quantityMost": {
                "type": "length",
                "rawValue": "6",
                "rawUnit": {
                    "name": "meters",
                    "type": "length",
                    "system": "SI base",
                    "offsetStart": 28,
                    "offsetEnd": 34,
                },
                "parsedValue": {
                    "name": "6",
                    "numeric": 6,
                    "structure": {"type": "NUMBER", "formatted": "6"},
                    "parsed": "6",
                },
                "normalizedQuantity": 6,
                "normalizedUnit": {"name": "m", "type": "length", "system": "SI base"},
                "offsetStart": 26,
                "offsetEnd": 27,
            },
        },
        {
            "type": "value",
            "quantity": {
                "type": "temperature",
                "rawValue": "5",
                "rawUnit": {
                    "name": "°C",
                    "type": "temperature",
                    "system": "SI derived",
                    "offsetStart": 56,
                    "offsetEnd": 58,
                },
                "parsedValue": {
                    "name": "5",
                    "numeric": 5,
                    "structure": {"type": "NUMBER", "formatted": "5"},
                    "parsed": "5",
                },
                "normalizedQuantity": 278.15,
                "normalizedUnit": {
                    "name": "K",
                    "type": "temperature",
                    "system": "SI base",
                },
                "offsetStart": 55,
                "offsetEnd": 56,
            },
        },
    ]

    basic_dependencies = [
        {
            "dep": "ROOT",
            "governor": 0,
            "governorGloss": "ROOT",
            "dependent": 9,
            "dependentGloss": "long",
        },
        {
            "dep": "compound",
            "governor": 2,
            "governorGloss": "text",
            "dependent": 1,
            "dependentGloss": "Example",
        },
        {
            "dep": "nsubj",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 2,
            "dependentGloss": "text",
        },
        {
            "dep": "cop",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 3,
            "dependentGloss": "is",
        },
        {
            "dep": "advmod",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 4,
            "dependentGloss": "between",
        },
        {
            "dep": "compound",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 5,
            "dependentGloss": "5",
        },
        {
            "dep": "punct",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 6,
            "dependentGloss": "-",
        },
        {
            "dep": "nummod",
            "governor": 8,
            "governorGloss": "meters",
            "dependent": 7,
            "dependentGloss": "6",
        },
        {
            "dep": "obl:npmod",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 8,
            "dependentGloss": "meters",
        },
        {
            "dep": "cc",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 10,
            "dependentGloss": "and",
        },
        {
            "dep": "det",
            "governor": 12,
            "governorGloss": "sun",
            "dependent": 11,
            "dependentGloss": "the",
        },
        {
            "dep": "nsubj",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 12,
            "dependentGloss": "sun",
        },
        {
            "dep": "cop",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 13,
            "dependentGloss": "is",
        },
        {
            "dep": "nummod",
            "governor": 15,
            "governorGloss": "°C",
            "dependent": 14,
            "dependentGloss": "5",
        },
        {
            "dep": "obl:npmod",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 15,
            "dependentGloss": "°C",
        },
        {
            "dep": "conj",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 16,
            "dependentGloss": "hot",
        },
        {
            "dep": "obl:tmod",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 17,
            "dependentGloss": "today",
        },
        {
            "dep": "punct",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 18,
            "dependentGloss": ".",
        },
    ]
    enhanced_dependencies = [
        {
            "dep": "ROOT",
            "governor": 0,
            "governorGloss": "ROOT",
            "dependent": 9,
            "dependentGloss": "long",
        },
        {
            "dep": "compound",
            "governor": 2,
            "governorGloss": "text",
            "dependent": 1,
            "dependentGloss": "Example",
        },
        {
            "dep": "nsubj",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 2,
            "dependentGloss": "text",
        },
        {
            "dep": "cop",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 3,
            "dependentGloss": "is",
        },
        {
            "dep": "advmod",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 4,
            "dependentGloss": "between",
        },
        {
            "dep": "compound",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 5,
            "dependentGloss": "5",
        },
        {
            "dep": "punct",
            "governor": 7,
            "governorGloss": "6",
            "dependent": 6,
            "dependentGloss": "-",
        },
        {
            "dep": "nummod",
            "governor": 8,
            "governorGloss": "meters",
            "dependent": 7,
            "dependentGloss": "6",
        },
        {
            "dep": "obl:npmod",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 8,
            "dependentGloss": "meters",
        },
        {
            "dep": "cc",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 10,
            "dependentGloss": "and",
        },
        {
            "dep": "det",
            "governor": 12,
            "governorGloss": "sun",
            "dependent": 11,
            "dependentGloss": "the",
        },
        {
            "dep": "nsubj",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 12,
            "dependentGloss": "sun",
        },
        {
            "dep": "cop",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 13,
            "dependentGloss": "is",
        },
        {
            "dep": "nummod",
            "governor": 15,
            "governorGloss": "°C",
            "dependent": 14,
            "dependentGloss": "5",
        },
        {
            "dep": "obl:npmod",
            "governor": 16,
            "governorGloss": "hot",
            "dependent": 15,
            "dependentGloss": "°C",
        },
        {
            "dep": "conj:and",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 16,
            "dependentGloss": "hot",
        },
        {
            "dep": "obl:tmod",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 17,
            "dependentGloss": "today",
        },
        {
            "dep": "punct",
            "governor": 9,
            "governorGloss": "long",
            "dependent": 18,
            "dependentGloss": ".",
        },
    ]
    enhanced_plus_plus_dependencies = deepcopy(enhanced_dependencies)
    tokens = [
        {
            "index": 1,
            "word": "Example",
            "originalText": "Example",
            "characterOffsetBegin": 0,
            "characterOffsetEnd": 7,
            "pos": "NN",
            "before": "",
            "after": " ",
        },
        {
            "index": 2,
            "word": "text",
            "originalText": "text",
            "characterOffsetBegin": 8,
            "characterOffsetEnd": 12,
            "pos": "NN",
            "before": " ",
            "after": " ",
        },
        {
            "index": 3,
            "word": "is",
            "originalText": "is",
            "characterOffsetBegin": 13,
            "characterOffsetEnd": 15,
            "pos": "VBZ",
            "before": " ",
            "after": " ",
        },
        {
            "index": 4,
            "word": "between",
            "originalText": "between",
            "characterOffsetBegin": 16,
            "characterOffsetEnd": 23,
            "pos": "IN",
            "before": " ",
            "after": " ",
        },
        {
            "index": 5,
            "word": "5",
            "originalText": "5",
            "characterOffsetBegin": 24,
            "characterOffsetEnd": 25,
            "pos": "CD",
            "before": " ",
            "after": "",
        },
        {
            "index": 6,
            "word": "-",
            "originalText": "-",
            "characterOffsetBegin": 25,
            "characterOffsetEnd": 26,
            "pos": "HYPH",
            "before": "",
            "after": "",
        },
        {
            "index": 7,
            "word": "6",
            "originalText": "6",
            "characterOffsetBegin": 26,
            "characterOffsetEnd": 27,
            "pos": "CD",
            "before": "",
            "after": " ",
        },
        {
            "index": 8,
            "word": "meters",
            "originalText": "meters",
            "characterOffsetBegin": 28,
            "characterOffsetEnd": 34,
            "pos": "NNS",
            "before": " ",
            "after": " ",
        },
        {
            "index": 9,
            "word": "long",
            "originalText": "long",
            "characterOffsetBegin": 35,
            "characterOffsetEnd": 39,
            "pos": "RB",
            "before": " ",
            "after": " ",
        },
        {
            "index": 10,
            "word": "and",
            "originalText": "and",
            "characterOffsetBegin": 40,
            "characterOffsetEnd": 43,
            "pos": "CC",
            "before": " ",
            "after": " ",
        },
        {
            "index": 11,
            "word": "the",
            "originalText": "the",
            "characterOffsetBegin": 44,
            "characterOffsetEnd": 47,
            "pos": "DT",
            "before": " ",
            "after": " ",
        },
        {
            "index": 12,
            "word": "sun",
            "originalText": "sun",
            "characterOffsetBegin": 48,
            "characterOffsetEnd": 51,
            "pos": "NN",
            "before": " ",
            "after": " ",
        },
        {
            "index": 13,
            "word": "is",
            "originalText": "is",
            "characterOffsetBegin": 52,
            "characterOffsetEnd": 54,
            "pos": "VBZ",
            "before": " ",
            "after": " ",
        },
        {
            "index": 14,
            "word": "5",
            "originalText": "5",
            "characterOffsetBegin": 55,
            "characterOffsetEnd": 56,
            "pos": "CD",
            "before": " ",
            "after": "",
        },
        {
            "index": 15,
            "word": "°C",
            "originalText": "°C",
            "characterOffsetBegin": 56,
            "characterOffsetEnd": 58,
            "pos": "NN",
            "before": "",
            "after": " ",
        },
        {
            "index": 16,
            "word": "hot",
            "originalText": "hot",
            "characterOffsetBegin": 59,
            "characterOffsetEnd": 62,
            "pos": "JJ",
            "before": " ",
            "after": " ",
        },
        {
            "index": 17,
            "word": "today",
            "originalText": "today",
            "characterOffsetBegin": 63,
            "characterOffsetEnd": 68,
            "pos": "NN",
            "before": " ",
            "after": "",
        },
        {
            "index": 18,
            "word": ".",
            "originalText": ".",
            "characterOffsetBegin": 68,
            "characterOffsetEnd": 69,
            "pos": ".",
            "before": "",
            "after": "",
        },
    ]
    core_nlp_response = {
        "sentences": [
            {
                "index": 0,
                "tokens": tokens,
                "basicDependencies": basic_dependencies,
                "enhancedDependencies": enhanced_dependencies,
                "enhancedPlusPlusDependencies": enhanced_plus_plus_dependencies,
            }
        ]
    }
    token_d = {token["index"]: token for token in tokens}
    results = {
        "text": text,
        "measurements": measurements,
        "core_nlp_response": core_nlp_response,
        "tokens": tokens,
        "token_d": token_d,
        "dependencies": basic_dependencies,
    }

    return results


@pytest.fixture(scope="session")
def interval_measurement():
    measurement = {
        "quantified": {
            "normalizedName": "known cases",
            "offsetEnd": 24,
            "offsetStart": 13,
            "rawName": "known cases",
        },
        "quantityLeast": {
            "normalizedQuantity": 0.7,
            "normalizedUnit": {"name": "one"},
            "offsetEnd": 392,
            "offsetStart": 390,
            "parsedValue": {
                "name": "70",
                "numeric": 70,
                "parsed": "70",
                "structure": {"formatted": "70", "type": "NUMBER"},
            },
            "rawUnit": {
                "name": "%",
                "offsetEnd": 396,
                "offsetStart": 395,
                "system": "non SI",
                "type": "fraction",
            },
            "rawValue": "70",
        },
        "quantityMost": {
            "normalizedQuantity": 0.8,
            "normalizedUnit": {"name": "one"},
            "offsetEnd": 395,
            "offsetStart": 393,
            "parsedValue": {
                "name": "80",
                "numeric": 80,
                "parsed": "80",
                "structure": {"formatted": "80", "type": "NUMBER"},
            },
            "rawUnit": {
                "name": "%",
                "offsetEnd": 396,
                "offsetStart": 395,
                "system": "non SI",
                "type": "fraction",
            },
            "rawValue": "80",
        },
        "type": "interval",
    }

    return measurement


@pytest.fixture(scope="session")
def extractor(model_entities):
    core_nlp_url = ""
    grobid_quantities_url = ""

    extractor = AttributeExtractor(
        core_nlp_url,
        grobid_quantities_url,
        model_entities,
    )

    return extractor


class TestAttributeExtraction:
    def test_get_quantity_type(self, extractor: AttributeExtractor):
        quantity = {"rawUnit": {"type": "mass"}}
        assert extractor.get_quantity_type(quantity) == "mass"

        quantity = {"normalizedUnit": {"type": "mass"}}
        assert extractor.get_quantity_type(quantity) == "mass"

        quantity = {
            "rawUnit": {"type": "raw_mass"},
            "normalizedUnit": {"type": "normalized_mass"},
        }
        assert extractor.get_quantity_type(quantity) == "raw_mass"

        quantity = {}
        assert extractor.get_quantity_type(quantity) == ""

    @pytest.mark.parametrize(
        ("measurement", "expected"),
        [
            ({"quantity": "q1"}, {"q1"}),
            ({"quantities": ["q1", "q2", "q3"]}, {"q1", "q2", "q3"}),
            ({"quantityMost": "q1", "quantityLeast": "q2"}, {"q1", "q2"}),
            ({"quantityBase": "q1", "quantityRange": "q2"}, {"q1", "q2"}),
        ],
    )
    def test_iter_quantities(
        self, extractor: AttributeExtractor, measurement, expected
    ):
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

    def test_iter_quantities_empty(self, extractor: AttributeExtractor):
        measurement: Dict[str, str] = {}
        expected: Set[str] = set()
        with pytest.warns(UserWarning) as warning_records:
            generated = set(extractor.iter_quantities(measurement))
        assert expected == generated
        assert len(warning_records) == 1
        assert warning_records[0].message.args[0] == "no quantity in measurement"

    @pytest.mark.parametrize(
        ("measurement", "expected_m_type"),
        [
            ({"quantity": {"rawUnit": {"type": "mass"}}}, "mass"),
            (
                {
                    "quantities": [
                        {"rawUnit": {"type": "mass"}},
                        {"rawUnit": {"type": "mass"}},
                        {"rawUnit": {"type": "fraction"}},
                        {"rawUnit": {"type": "fraction"}},
                        {"rawUnit": {"type": "fraction"}},
                    ]
                },
                "fraction",
            ),
            (
                {
                    "quantities": [
                        {"rawUnit": {"type": ""}},
                        {"rawUnit": {"type": ""}},
                        {"rawUnit": {"type": "mass"}},
                        {"rawUnit": {"type": "mass"}},
                    ]
                },
                "mass",
            ),
        ],
    )
    def test_get_measurement_type(
        self, extractor: AttributeExtractor, measurement, expected_m_type
    ):
        m_type = extractor.get_measurement_type(measurement)
        assert m_type == expected_m_type

    def test_count_measurement_types(self, extractor: AttributeExtractor):
        measurements = [
            {"quantity": {"rawUnit": {"type": "mass"}}},
            {"quantity": {"rawUnit": {"type": "mass"}}},
            {"quantity": {"rawUnit": {"type": "mass"}}},
            {"quantity": {"rawUnit": {"type": "fraction"}}},
            {"quantity": {"rawUnit": {"type": "fraction"}}},
        ]
        counts = extractor.count_measurement_types(measurements)
        assert counts == {"mass": 3, "fraction": 2}

    def test_get_overlapping_token_ids(self, extractor: AttributeExtractor):
        tokens = [
            {"index": 0, "characterOffsetBegin": 0, "characterOffsetEnd": 2},
            {"index": 1, "characterOffsetBegin": 2, "characterOffsetEnd": 5},
            {"index": 2, "characterOffsetBegin": 5, "characterOffsetEnd": 10},
            {"index": 3, "characterOffsetBegin": 10, "characterOffsetEnd": 12},
        ]

        ids = extractor.get_overlapping_token_ids(0, 2, tokens)
        assert set(ids) == {0}

        ids = extractor.get_overlapping_token_ids(0, 5, tokens)
        assert set(ids) == {0, 1}

        ids = extractor.get_overlapping_token_ids(6, 8, tokens)
        assert set(ids) == {2}

        ids = extractor.get_overlapping_token_ids(6, 13, tokens)
        assert set(ids) == {2, 3}

    def test_get_grobid_measurements(self, extractor: AttributeExtractor, monkeypatch):
        text = "Example text is 5m long."
        real_measurements = [
            {
                "type": "value",
                "quantity": {
                    "type": "length",
                    "rawValue": "5",
                    "rawUnit": {
                        "name": "m",
                        "type": "length",
                        "system": "SI base",
                        "offsetStart": 17,
                        "offsetEnd": 18,
                    },
                    "parsedValue": {
                        "name": "5",
                        "numeric": 5,
                        "structure": {"type": "NUMBER", "formatted": "5"},
                        "parsed": "5",
                    },
                    "normalizedQuantity": 5,
                    "normalizedUnit": {
                        "name": "m",
                        "type": "length",
                        "system": "SI base",
                    },
                    "offsetStart": 16,
                    "offsetEnd": 17,
                },
            }
        ]
        response_json = {"measurements": real_measurements}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(response_json).encode("utf-8")

        fake_requests = Mock()
        monkeypatch.setattr("bluesearch.mining.attribute.requests", fake_requests)
        fake_requests.post.return_value = response

        # Test 1
        _ = extractor.get_grobid_measurements(text)
        fake_requests.post.assert_called_once()

        # Test 2
        response.status_code = 500
        with pytest.warns(UserWarning) as warning_records:
            measurements = extractor.get_grobid_measurements(text)
        assert len(measurements) == 0
        assert len(warning_records) == 1
        assert (
            warning_records[0].message.args[0]
            == f"GROBID request problem. Code: {response.status_code}"
        )

        # Test 3
        response.status_code = 200
        response._content = json.dumps({}).encode("utf-8")
        measurements = extractor.get_grobid_measurements(text)
        assert len(measurements) == 0

    def test_annotate_quantities(self, extractor: AttributeExtractor):
        text = "Example text is 5m long."
        measurements = [
            {
                "type": "value",
                "quantity": {
                    "type": "length",
                    "rawValue": "5",
                    "rawUnit": {
                        "name": "m",
                        "type": "length",
                        "system": "SI base",
                        "offsetStart": 17,
                        "offsetEnd": 18,
                    },
                    "parsedValue": {
                        "name": "5",
                        "numeric": 5,
                        "structure": {"type": "NUMBER", "formatted": "5"},
                        "parsed": "5",
                    },
                    "normalizedQuantity": 5,
                    "normalizedUnit": {
                        "name": "m",
                        "type": "length",
                        "system": "SI base",
                    },
                    "offsetStart": 16,
                    "offsetEnd": 17,
                },
            }
        ]

        annotated_text = extractor.annotate_quantities(text, measurements)
        assert isinstance(annotated_text, HTML)

    def test_get_quantity_tokens(self, extractor: AttributeExtractor):
        # text = "Example text is 5 m long."
        quantity = {
            "type": "length",
            "rawValue": "5",
            "rawUnit": {
                "name": "m",
                "type": "length",
                "system": "SI base",
                "offsetStart": 18,
                "offsetEnd": 19,
            },
            "parsedValue": {
                "name": "5",
                "numeric": 5,
                "structure": {"type": "NUMBER", "formatted": "5"},
                "parsed": "5",
            },
            "normalizedQuantity": 5,
            "normalizedUnit": {"name": "m", "type": "length", "system": "SI base"},
            "offsetStart": 16,
            "offsetEnd": 17,
        }
        tokens = [
            {
                "index": 1,
                "word": "Example",
                "originalText": "Example",
                "characterOffsetBegin": 0,
                "characterOffsetEnd": 7,
                "pos": "NN",
                "before": "",
                "after": " ",
            },
            {
                "index": 2,
                "word": "text",
                "originalText": "text",
                "characterOffsetBegin": 8,
                "characterOffsetEnd": 12,
                "pos": "NN",
                "before": " ",
                "after": " ",
            },
            {
                "index": 3,
                "word": "is",
                "originalText": "is",
                "characterOffsetBegin": 13,
                "characterOffsetEnd": 15,
                "pos": "VBZ",
                "before": " ",
                "after": " ",
            },
            {
                "index": 4,
                "word": "5",
                "originalText": "5",
                "characterOffsetBegin": 16,
                "characterOffsetEnd": 17,
                "pos": "CD",
                "before": " ",
                "after": " ",
            },
            {
                "index": 5,
                "word": "m",
                "originalText": "m",
                "characterOffsetBegin": 18,
                "characterOffsetEnd": 19,
                "pos": "NNS",
                "before": " ",
                "after": " ",
            },
            {
                "index": 6,
                "word": "long",
                "originalText": "long",
                "characterOffsetBegin": 20,
                "characterOffsetEnd": 24,
                "pos": "RB",
                "before": " ",
                "after": "",
            },
            {
                "index": 7,
                "word": ".",
                "originalText": ".",
                "characterOffsetBegin": 24,
                "characterOffsetEnd": 25,
                "pos": ".",
                "before": "",
                "after": "",
            },
        ]

        found_tokens = extractor.get_quantity_tokens(quantity, tokens)
        assert len(found_tokens) == 2
        assert set(found_tokens) == {4, 5}

    def test_get_measurement_tokens(self, extractor, example_results):
        measurement = example_results["measurements"][0]
        tokens = example_results["tokens"]

        measurement_tokens = extractor.get_measurement_tokens(measurement, tokens)
        assert set(measurement_tokens) == {5, 7, 8}

    def test_get_entity_tokens(self, model_entities, extractor, example_results):
        text = "Example text is between 5-6 meters long."
        doc = model_entities(text)
        entity = doc[2:4]
        tokens = example_results["tokens"]

        entity_tokens = extractor.get_entity_tokens(entity, tokens)
        assert set(entity_tokens) == {3, 4}

    def test_iter_parents(self, extractor, example_results):
        dependencies = example_results["dependencies"]

        child_idx = 1
        parent_ids = extractor.iter_parents(dependencies, child_idx)
        assert set(parent_ids) == {2}

        child_idx = 9
        parent_ids = extractor.iter_parents(dependencies, child_idx)
        assert set(parent_ids) == set()

    def test_find_nn_parents(self, extractor, example_results):
        dependencies = example_results["dependencies"]
        token_d = example_results["token_d"]

        token_idx = 4
        assert token_d[token_idx]["word"] == "between"
        parent_ids = extractor.find_nn_parents(dependencies, token_d, token_idx)
        assert set(parent_ids) == {8}

        token_idx = 1
        assert token_d[token_idx]["word"] == "Example"
        parent_ids = extractor.find_nn_parents(dependencies, token_d, token_idx)
        assert set(parent_ids) == {1, 2}

    def test_find_all_parents(self, extractor, example_results):
        dependencies = example_results["dependencies"]
        token_d = example_results["token_d"]

        token_ids = [2, 3, 4]
        parent_ids = extractor.find_all_parents(dependencies, token_d, token_ids)
        assert set(parent_ids) == {2, 8}

    def test_quantity_to_str(self, extractor, example_results):
        measurement = example_results["measurements"][0]
        quantity_1 = extractor.quantity_to_str(measurement["quantityLeast"])
        quantity_2 = extractor.quantity_to_str(measurement["quantityMost"])

        assert quantity_1 == "5 meters"
        assert quantity_2 == "6 meters"

    def test_measurement_to_str(self, extractor, example_results):
        measurement = example_results["measurements"][0]
        measurement_str = extractor.measurement_to_str(measurement)

        assert measurement_str == ["5 meters", "6 meters"]

    def test_process_raw_annotation_df(self, extractor, example_results):
        df = pd.DataFrame()
        df_processed = extractor.process_raw_annotation_df(df, copy=False)
        assert df is df_processed

        measurement = example_results["measurements"][0]
        df = pd.DataFrame([{"attribute": measurement}])
        df_processed = extractor.process_raw_annotation_df(df)
        row = df_processed.iloc[0]
        assert row["property"] == "has length interval"
        assert row["property_type"] == "attribute"
        assert row["property_value"] == ["5 meters", "6 meters"]
        assert row["property_value_type"] == "int"

    def test_get_core_nlp_analysis(self, extractor, example_results, monkeypatch):
        text = example_results["text"]

        core_nlp_response = example_results["core_nlp_response"]
        real_response_json = core_nlp_response
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(real_response_json).encode("utf-8")

        fake_requests = Mock()
        monkeypatch.setattr("bluesearch.mining.attribute.requests", fake_requests)
        fake_requests.post.return_value = response

        response_json = extractor.get_core_nlp_analysis(text)
        assert response_json.keys() == real_response_json.keys()

    def test_are_linked(self, model_entities, extractor, example_results):
        core_nlp_response = example_results["core_nlp_response"]
        core_nlp_sentence = core_nlp_response["sentences"][0]
        measurement = example_results["measurements"][0]
        text = example_results["text"]
        doc = model_entities(text)

        expect = [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        for i, expected_result in enumerate(expect):
            result = extractor.are_linked(
                measurement, doc[i : i + 1], core_nlp_sentence
            )
            assert result == expected_result

    def test_extract_attributes(self, extractor, example_results, monkeypatch):
        text = example_results["text"]

        measurements = example_results["measurements"]
        core_nlp_response = example_results["core_nlp_response"]

        def fake_grobid(text):
            return measurements

        def fake_core_nlp(text):
            return core_nlp_response

        monkeypatch.setattr(extractor, "get_grobid_measurements", fake_grobid)
        monkeypatch.setattr(extractor, "get_core_nlp_analysis", fake_core_nlp)

        expect_columns = [
            "property",
            "property_type",
            "property_value",
            "property_value_type",
        ]

        df = extractor.extract_attributes(text)
        assert isinstance(df, pd.DataFrame)
        assert set(expect_columns).issubset(set(df.columns))

        extractor.ee_model = spacy.load("en_core_web_sm")
        extractor.ee_model.remove_pipe("ner")

        df = extractor.extract_attributes(text)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert set(expect_columns).issubset(set(df.columns))

        df = extractor.extract_attributes(text, linked_attributes_only=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(expect_columns).issubset(set(df.columns))

        df = extractor.extract_attributes(text, raw_attributes=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert {"attribute"}.issubset(set(df.columns))


class TestAttributeAnnotationTab:
    def test_init(self, extractor, model_entities):
        annotation_tab = AttributeAnnotationTab(extractor, model_entities)
        assert isinstance(annotation_tab, AttributeAnnotationTab)

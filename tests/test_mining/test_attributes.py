import json
import pytest
from unittest.mock import Mock

from IPython.display import HTML
import requests
import spacy

from bbsearch.mining import AttributeExtractor


@pytest.fixture(scope='session')
def model_entities():
    """Standard English spacy model.

    References
    ----------
    https://spacy.io/api/annotation#named-entities
    """
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="session")
def interval_measurement():
    measurement = {
        'quantified': {'normalizedName': 'known cases',
                       'offsetEnd': 24,
                       'offsetStart': 13,
                       'rawName': 'known cases'},
        'quantityLeast': {'normalizedQuantity': 0.7,
                          'normalizedUnit': {'name': 'one'},
                          'offsetEnd': 392,
                          'offsetStart': 390,
                          'parsedValue': {'name': '70',
                                          'numeric': 70,
                                          'parsed': '70',
                                          'structure': {'formatted': '70',
                                                        'type': 'NUMBER'}},
                          'rawUnit': {'name': '%',
                                      'offsetEnd': 396,
                                      'offsetStart': 395,
                                      'system': 'non SI',
                                      'type': 'fraction'},
                          'rawValue': '70'},
        'quantityMost': {'normalizedQuantity': 0.8,
                         'normalizedUnit': {'name': 'one'},
                         'offsetEnd': 395,
                         'offsetStart': 393,
                         'parsedValue': {'name': '80',
                                         'numeric': 80,
                                         'parsed': '80',
                                         'structure': {'formatted': '80',
                                                       'type': 'NUMBER'}},
                         'rawUnit': {'name': '%',
                                     'offsetEnd': 396,
                                     'offsetStart': 395,
                                     'system': 'non SI',
                                     'type': 'fraction'},
                         'rawValue': '80'},
        'type': 'interval'}

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
            "normalizedUnit": {"type": "normalized_mass"}}
        assert extractor.get_quantity_type(quantity) == "raw_mass"

        quantity = {}
        assert extractor.get_quantity_type(quantity) == ""

    def test_iter_quantities(self, extractor: AttributeExtractor):
        measurement = {"quantity": "q1"}
        expected = {"q1"}
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

        measurement = {"quantities": ["q1", "q2", "q3"]}
        expected = {"q1", "q2", "q3"}
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

        measurement = {"quantityMost": "q1", "quantityLeast": "q2"}
        expected = {"q1", "q2"}
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

        measurement = {"quantityBase": "q1", "quantityRange": "q2"}
        expected = {"q1", "q2"}
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

        measurement = {}
        expected = set()
        generated = set(extractor.iter_quantities(measurement))
        assert expected == generated

    def test_get_measurement_type(self, extractor: AttributeExtractor):
        measurement = {"quantity": {"rawUnit": {"type": "mass"}}}
        m_type = extractor.get_measurement_type(measurement)
        assert m_type == "mass"

        measurement = {"quantities": [
            {"rawUnit": {"type": "mass"}},
            {"rawUnit": {"type": "mass"}},
            {"rawUnit": {"type": "fraction"}},
            {"rawUnit": {"type": "fraction"}},
            {"rawUnit": {"type": "fraction"}},
        ]}
        m_type = extractor.get_measurement_type(measurement)
        assert m_type == "fraction"

        measurement = {"quantities": [
            {"rawUnit": {"type": ""}},
            {"rawUnit": {"type": ""}},
            {"rawUnit": {"type": "mass"}},
            {"rawUnit": {"type": "mass"}},
        ]}
        m_type = extractor.get_measurement_type(measurement)
        assert m_type == "mass"

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
            {'type': 'value',
             'quantity': {'type': 'length',
                          'rawValue': '5',
                          'rawUnit': {'name': 'm',
                                      'type': 'length',
                                      'system': 'SI base',
                                      'offsetStart': 17,
                                      'offsetEnd': 18},
                          'parsedValue': {'name': '5',
                                          'numeric': 5,
                                          'structure': {'type': 'NUMBER', 'formatted': '5'},
                                          'parsed': '5'},
                          'normalizedQuantity': 5,
                          'normalizedUnit': {'name': 'm', 'type': 'length', 'system': 'SI base'},
                          'offsetStart': 16,
                          'offsetEnd': 17}}]
        response_json = {"measurements": real_measurements}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(response_json).encode("utf-8")

        fake_requests = Mock()
        monkeypatch.setattr('bbsearch.mining.attributes.requests', fake_requests)
        fake_requests.post.return_value = response

        # Test 1
        measurements = extractor.get_grobid_measurements(text)
        fake_requests.post.assert_called_once()

        # Test 2
        response.status_code = 500
        measurements = extractor.get_grobid_measurements(text)
        assert len(measurements) == 0

        # Test 3
        response.status_code = 200
        response._content = json.dumps({}).encode("utf-8")
        measurements = extractor.get_grobid_measurements(text)
        assert len(measurements) == 0

    def test_annotate_quantities(self, extractor: AttributeExtractor):
        text = "Example text is 5m long."
        measurements = [
            {'type': 'value',
             'quantity': {'type': 'length',
                          'rawValue': '5',
                          'rawUnit': {'name': 'm',
                                      'type': 'length',
                                      'system': 'SI base',
                                      'offsetStart': 17,
                                      'offsetEnd': 18},
                          'parsedValue': {'name': '5',
                                          'numeric': 5,
                                          'structure': {'type': 'NUMBER', 'formatted': '5'},
                                          'parsed': '5'},
                          'normalizedQuantity': 5,
                          'normalizedUnit': {'name': 'm', 'type': 'length', 'system': 'SI base'},
                          'offsetStart': 16,
                          'offsetEnd': 17}}]

        annotated_text = extractor.annotate_quantities(text, measurements, width=70)
        assert isinstance(annotated_text, HTML)

    def test_get_quantity_tokens(self, extractor: AttributeExtractor):
        text = "Example text is 5 m long."
        quantity = {
            'type': 'length',
            'rawValue': '5',
            'rawUnit': {'name': 'm',
                        'type': 'length',
                        'system': 'SI base',
                        'offsetStart': 18,
                        'offsetEnd': 19},
            'parsedValue': {'name': '5',
                            'numeric': 5,
                            'structure': {'type': 'NUMBER', 'formatted': '5'},
                            'parsed': '5'},
            'normalizedQuantity': 5,
            'normalizedUnit': {'name': 'm', 'type': 'length', 'system': 'SI base'},
            'offsetStart': 16,
            'offsetEnd': 17}
        tokens = [
            {'index': 1,
             'word': 'Example',
             'originalText': 'Example',
             'characterOffsetBegin': 0,
             'characterOffsetEnd': 7,
             'pos': 'NN',
             'before': '',
             'after': ' '},
            {'index': 2,
             'word': 'text',
             'originalText': 'text',
             'characterOffsetBegin': 8,
             'characterOffsetEnd': 12,
             'pos': 'NN',
             'before': ' ',
             'after': ' '},
            {'index': 3,
             'word': 'is',
             'originalText': 'is',
             'characterOffsetBegin': 13,
             'characterOffsetEnd': 15,
             'pos': 'VBZ',
             'before': ' ',
             'after': ' '},
            {'index': 4,
             'word': '5',
             'originalText': '5',
             'characterOffsetBegin': 16,
             'characterOffsetEnd': 17,
             'pos': 'CD',
             'before': ' ',
             'after': ' '},
            {'index': 5,
             'word': 'm',
             'originalText': 'm',
             'characterOffsetBegin': 18,
             'characterOffsetEnd': 19,
             'pos': 'NNS',
             'before': ' ',
             'after': ' '},
            {'index': 6,
             'word': 'long',
             'originalText': 'long',
             'characterOffsetBegin': 20,
             'characterOffsetEnd': 24,
             'pos': 'RB',
             'before': ' ',
             'after': ''},
            {'index': 7,
             'word': '.',
             'originalText': '.',
             'characterOffsetBegin': 24,
             'characterOffsetEnd': 25,
             'pos': '.',
             'before': '',
             'after': ''}]

        found_tokens = extractor.get_quantity_tokens(quantity, tokens)
        assert len(found_tokens) == 2
        assert set(found_tokens) == {4, 5}

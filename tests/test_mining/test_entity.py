"""Collections of tests covering the `entity.py` module."""
import pathlib

from bbsearch.mining import dump_jsonl, load_jsonl, remap_entity_type


def test_load_jsonl(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "file.jsonl"

    li = [{"a": 1, "b": "cc"}, {"k": 23}]
    dump_jsonl(li, path)
    lo = load_jsonl(path)

    assert li == lo


def test_entity_type():
    patterns = [
        {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
        {"label": "DISEASE", "pattern": [{"LOWER": "covid"}, {"TEXT": '-'}, {"TEXT": "19"}]},
        {"label": "CHEMICAL", "pattern": [{"LOWER": "glucose"}]},
    ]

    etype_mapping = {"CHEMICAL": "CHEBI"}

    adjusted_patterns = remap_entity_type(patterns, etype_mapping)
    adjusted_patterns_true = [
        {"label": "NaE", "pattern": [{"LOWER": "covid-19"}]},
        {"label": "NaE", "pattern": [{"LOWER": "covid"}, {"TEXT": '-'}, {"TEXT": "19"}]},
        {"label": "CHEBI", "pattern": [{"LOWER": "glucose"}]}]

    assert patterns is not adjusted_patterns
    assert adjusted_patterns == adjusted_patterns_true

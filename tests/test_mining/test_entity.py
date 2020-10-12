"""Collections of tests covering the `entity.py` module."""
import pathlib

from bbsearch.mining import dump_jsonl, load_jsonl


def test_load_jsonl(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "file.jsonl"

    li = [{"a": 1, "b": "cc"}, {"k": 23}]
    dump_jsonl(li, path)
    lo = load_jsonl(path)

    assert li == li

"""Collection of tests that make sure that fixtures are set up correctly.

Notes
-----
The internals of fixtures might vary based on how conftest.py sets them up. The goal of these
tests is to run simple sanity checks rather than detailed bookkeeping.

"""
from sqlite3 import OperationalError

import numpy as np
import pandas as pd
import pytest


def test_database(fake_db_cursor):
    """Make sure database tables setup correctly."""
    for table_name in ['articles', 'article_id_2_sha', 'sentences']:
        res = fake_db_cursor.execute('SELECT * FROM {}'.format(table_name)).fetchall()

        assert len(res) > 0

    with pytest.raises(OperationalError):
        fake_db_cursor.execute('SELECT * FROM fake_table').fetchall()


def test_embeddings(embeddings_path, fake_db_cursor):
    """Make sure all sentences are embedded."""
    n_sentences = fake_db_cursor.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]

    for p in embeddings_path.iterdir():
        a = np.load(str(p))

        assert isinstance(a, np.ndarray)
        assert a.shape[0] == n_sentences
        assert a.shape[1] > 0


def test_h5(embeddings_h5_path):
    assert embeddings_h5_path.is_file()


def test_metadata(metadata_path):
    """Make sure all metadata csv is correct"""
    df = pd.read_csv(str(metadata_path))

    assert len(df) > 0


def test_jsons(jsons_path):
    """Make sure all jsons are present."""
    n_json_files = len(list(jsons_path.rglob('*.json')))

    assert n_json_files > 0

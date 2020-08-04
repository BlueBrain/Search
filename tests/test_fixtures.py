"""Collection of tests that make sure that fixtures are set up correctly.

Notes
-----
The internals of fixtures might vary based on how conftest.py sets them up. The goal of these
tests is to run simple sanity checks rather than detailed bookkeeping.

"""
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError


def test_database(fake_sqlalchemy_engine):
    """Make sure database tables setup correctly."""
    for table_name in ['articles', 'sentences']:
        res = pd.read_sql('SELECT * FROM {}'.format(table_name), fake_sqlalchemy_engine)

        assert len(res) > 0

    with pytest.raises(OperationalError):
        fake_sqlalchemy_engine.execute('SELECT * FROM fake_table').all()


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

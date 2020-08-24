"""Collection of tests that make sure that fixtures are set up correctly.

Notes
-----
The internals of fixtures might vary based on how conftest.py sets them up. The goal of these
tests is to run simple sanity checks rather than detailed bookkeeping.

"""
import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy.exc import OperationalError, ProgrammingError


def test_database(fake_sqlalchemy_engine, backend_database):
    """Make sure database tables setup correctly."""
    inspector = sqlalchemy.inspect(fake_sqlalchemy_engine)

    for table_name in ['articles', 'sentences', 'mining_cache']:
        res = pd.read_sql('SELECT * FROM {}'.format(table_name), fake_sqlalchemy_engine)

        if table_name != 'articles':
            # Mysql consider that sentences table has 2 indexes (article_id one + UNIQUE
            # constraint)
            # sqlite will only consider 1 index for this table (article_id one)
            assert len(inspector.get_indexes(table_name)) >= 1

        assert len(res) > 0
    if backend_database == 'sqlite':
        with pytest.raises(OperationalError):
            fake_sqlalchemy_engine.execute('SELECT * FROM fake_table').all()
    else:
        with pytest.raises(ProgrammingError):
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

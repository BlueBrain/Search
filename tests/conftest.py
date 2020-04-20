from pathlib import Path
import sqlite3

import numpy as np
import pytest

ROOT_PATH = Path(__file__).resolve().parent.parent  # root of the repository


@pytest.fixture(scope='session')
def assets_path(tmp_path_factory):
    """Path to assets."""
    assets_path = tmp_path_factory.mktemp('assets', numbered=False)

    return assets_path


@pytest.fixture(scope='session')
def cnxn(tmp_path_factory):
    """Connection object (sqlite)."""
    db_path = tmp_path_factory.mktemp('db', numbered=False) / 'dummy.sqlite'
    cnxn = sqlite3.connect(str(db_path))
    yield cnxn

    cnxn.close()  # disconnect


@pytest.fixture(scope='session')
def cursor(cnxn, jsons_path, metadata_path):
    """Database object (sqlite)."""
    cursor = cnxn.cursor()
    # create
    stmt_create_articles = """CREATE TABLE articles (article_id TEXT PRIMARY KEY, publisher TEXT)"""
    stmt_create_sentences = """CREATE TABLE sentences (sentence_id INTEGER PRIMARY KEY, text TEXT)"""
    cursor.execute(stmt_create_articles)
    cursor.execute(stmt_create_sentences)

    # Populate
    cursor.executemany("INSERT INTO sentences (sentence_id, text) values (?, ?)",
                       ((i, 'whatever') for i in range(2000)))

    yield cursor

    cnxn.rollback()  # undo uncommited changes -> after tests are run all changes are deleted INVESTIGATE


@pytest.fixture(scope='session')
def jsons_path():
    """Path to a directory where jsons are stored."""
    jsons_path = ROOT_PATH / 'tests' / 'data' / 'CORD19_samples'
    assert jsons_path.exists()

    return jsons_path


@pytest.fixture(scope='session')
def metadata_path():
    """Path to metadata.csv."""
    metadata_path = ROOT_PATH / 'tests' / 'data' / 'CORD19_samples' / 'metadata.csv'
    assert metadata_path.exists()

    return metadata_path


@pytest.fixture(scope='session')
def embeddings_path(tmp_path_factory, cursor):
    """Path to a directory where emebddings stored."""
    random_state = 3
    np.random.seed(random_state)
    models = ['BERT', 'BIOBERT']

    n_sentences = cursor.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
    embeddings_path = tmp_path_factory.mktemp('embeddings', numbered=False)
    for model in models:
        model_path = embeddings_path / model / '{}.npy'.format(model)
        model_path.parent.mkdir(parents=True)
        np.save(str(model_path), np.random.rand(n_sentences, 2))

    return embeddings_path

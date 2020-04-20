"""Collection of tests that make sure that fixtures are set up correctly."""

import numpy as np
import pandas as pd


def test_sentences(cursor):
    """Make sure sentences table setup correctly."""
    all = cursor.execute('SELECT * FROM sentences').fetchall()

    assert len(all) == 2000


def test_embeddings(embeddings_path):
    """Make sure all sentences are embedded."""
    for p in embeddings_path.iterdir():
        model_path = p / '{}.npy'.format(p.stem)

        a = np.load(str(model_path))

        assert a.shape == (2000, 2)


def test_metadata(metadata_path):
    """Make sure all metadata csv is correct"""
    df = pd.read_csv(str(metadata_path))

    assert len(df) == 11


def test_jsons(jsons_path):
    """Make sure all jsons are present."""
    n_json_files = len(list(jsons_path.glob('**/*json')))

    assert n_json_files == 17

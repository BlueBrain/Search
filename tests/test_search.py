import pathlib
from unittest.mock import Mock

import numpy as np
import pytest

from bbsearch.search import LocalSearcher, run_search


class TestLocalSearcher:
    def test_error(self, tmpdir):
        fake_dir = pathlib.Path(str(tmpdir))

        with pytest.raises(FileNotFoundError):
            LocalSearcher({}, {}, fake_dir)

    def test_query(self, embeddings_path, tmp_path_factory):
        emb_path = [p for p in embeddings_path.iterdir()][0]

        model_name = emb_path.stem
        query_text = 'some text'
        k = 3

        # load embeddings
        precomputed_embeddings = {model_name: np.load(str(emb_path))}
        dim = precomputed_embeddings[model_name].shape[1] - 1

        # fake embedding model
        emb_mod = Mock()
        emb_mod.preprocess.return_value = query_text
        emb_mod.embed.return_value = np.ones((dim,))
        embedding_models = {model_name: emb_mod}

        # get database path
        database_path = tmp_path_factory.getbasetemp() / 'db' / 'cord19.db'  # created by fake_db_cnxn fixture

        # actual test
        local_searcher = LocalSearcher(embedding_models, precomputed_embeddings, database_path)
        indices, similarities, stats = local_searcher.query(model_name, k=k, query_text=query_text)

        assert indices.shape == (k,)
        assert similarities.shape == (k,)
        assert isinstance(stats, dict)
        assert emb_mod.preprocess.call_count == 1
        assert emb_mod.embed.call_count == 1


def test_run_search(fake_db_cnxn, embeddings_path):
    model = 'SBERT'
    query_text = 'I want to know everything about the universe.'
    k = 5

    emb_mod = Mock()
    emb_mod.preprocess.return_value = query_text
    emb_mod.embed.return_value = np.ones((2,))

    precomputed_embeddings = np.load(str(embeddings_path / f'{model}.npy'))

    deprioritized_text = 'Vegetables'
    deprioritized_strength = 'Mild'

    exclusion_text = """dshdasihsdakjbsadhbasoiasdhkjbad \n
                     hasdn,asdnioahsdpihapsdipas;kdn \n
                     jpaijdspojasdn opjpo"""

    indices, similarities, stats = run_search(embedding_model=emb_mod,
                                              precomputed_embeddings=precomputed_embeddings,
                                              database=fake_db_cnxn,
                                              query_text=query_text,
                                              deprioritize_text=None,
                                              deprioritize_strength=deprioritized_strength,
                                              has_journal=True,
                                              date_range=(2000, 2021),
                                              exclusion_text=exclusion_text,
                                              k=k)

    assert indices.shape == (k,)
    assert similarities.shape == (k,)
    assert isinstance(stats, dict)
    assert emb_mod.preprocess.call_count == 1
    assert emb_mod.embed.call_count == 1

    indices, similarities, stats = run_search(embedding_model=emb_mod,
                                              precomputed_embeddings=precomputed_embeddings,
                                              database=fake_db_cnxn,
                                              query_text=query_text,
                                              deprioritize_text=deprioritized_text,
                                              date_range=(3000, 3001),
                                              k=k)

    assert indices.shape == (0,)
    assert similarities.shape == (0,)
    assert isinstance(stats, dict)
    assert emb_mod.preprocess.call_count == 3
    assert emb_mod.embed.call_count == 3

    indices, similarities, stats = run_search(embedding_model=emb_mod,
                                              precomputed_embeddings=precomputed_embeddings,
                                              database=fake_db_cnxn,
                                              query_text=query_text,
                                              deprioritize_text=deprioritized_text,
                                              k=k)

    assert indices.shape == (k,)
    assert similarities.shape == (k,)
    assert isinstance(stats, dict)

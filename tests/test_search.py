from unittest.mock import Mock

import h5py
import numpy as np

from bbsearch.search import LocalSearcher, run_search
from bbsearch.utils import H5


class TestLocalSearcher:

    def test_query(self, embeddings_h5_path, fake_sqlalchemy_engine):
        with h5py.File(embeddings_h5_path, 'r') as f:
            model_name = list(f.keys())[0]

        query_text = 'some text'
        k = 3

        # only take populated rows
        indices = H5.find_populated_rows(embeddings_h5_path, model_name)
        embeddings = H5.load(embeddings_h5_path, model_name, indices=indices)

        # load embeddings
        precomputed_embeddings = {model_name: embeddings}
        dim = precomputed_embeddings[model_name].shape[1]

        # fake embedding model
        emb_mod = Mock()
        emb_mod.preprocess.return_value = query_text
        emb_mod.embed.return_value = np.ones((dim,))
        embedding_models = {model_name: emb_mod}

        # actual test
        local_searcher = LocalSearcher(embedding_models,
                                       precomputed_embeddings,
                                       indices,
                                       fake_sqlalchemy_engine)
        top_indices, similarities, stats = local_searcher.query(model_name,
                                                                k=k,
                                                                query_text=query_text)

        assert top_indices.shape == (k,)
        assert similarities.shape == (k,)
        assert isinstance(stats, dict)
        assert emb_mod.preprocess.call_count == 1
        assert emb_mod.embed.call_count == 1


def test_run_search(fake_sqlalchemy_engine, embeddings_h5_path):
    model = 'SBERT'
    query_text = 'I want to know everything about the universe.'
    k = 5

    emb_mod = Mock()
    emb_mod.preprocess.return_value = query_text
    emb_mod.embed.return_value = np.ones((2,))

    # only take populated rows
    indices = H5.find_populated_rows(embeddings_h5_path, model)
    precomputed_embeddings = H5.load(embeddings_h5_path, model, indices=indices)

    deprioritized_text = 'Vegetables'
    deprioritized_strength = 'Mild'

    exclusion_text = """dshdasihsdakjbsadhbasoiasdhkjbad \n
                     hasdn,asdnioahsdpihapsdipas;kdn \n
                     jpaijdspojasdn opjpo"""

    top_indices, similarities, stats = run_search(embedding_model=emb_mod,
                                                  precomputed_embeddings=precomputed_embeddings,
                                                  indices=indices,
                                                  connection=fake_sqlalchemy_engine,
                                                  query_text=query_text,
                                                  deprioritize_text=None,
                                                  deprioritize_strength=deprioritized_strength,
                                                  has_journal=True,
                                                  date_range=(2000, 2021),
                                                  exclusion_text=exclusion_text,
                                                  k=k)

    assert top_indices.shape == (k,)
    assert similarities.shape == (k,)
    assert isinstance(stats, dict)
    assert emb_mod.preprocess.call_count == 1
    assert emb_mod.embed.call_count == 1

    top_indices, similarities, stats = run_search(embedding_model=emb_mod,
                                                  precomputed_embeddings=precomputed_embeddings,
                                                  indices=indices,
                                                  connection=fake_sqlalchemy_engine,
                                                  query_text=query_text,
                                                  deprioritize_text=deprioritized_text,
                                                  date_range=(3000, 3001),
                                                  k=k)

    assert top_indices.shape == (0,)
    assert similarities.shape == (0,)
    assert isinstance(stats, dict)
    assert emb_mod.preprocess.call_count == 2
    assert emb_mod.embed.call_count == 2

    top_indices, similarities, stats = run_search(embedding_model=emb_mod,
                                                  precomputed_embeddings=precomputed_embeddings,
                                                  indices=indices,
                                                  connection=fake_sqlalchemy_engine,
                                                  query_text=query_text,
                                                  deprioritize_text=deprioritized_text,
                                                  k=k)

    assert top_indices.shape == (k,)
    assert similarities.shape == (k,)
    assert isinstance(stats, dict)

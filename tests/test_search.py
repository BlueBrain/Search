from unittest.mock import Mock

import numpy as np
import pytest
import torch

from bbsearch.search import SearchEngine
from bbsearch.utils import H5


class TestSearchEngine:

    @pytest.mark.parametrize('granularity', ['sentences', 'articles'])
    def test_run_search(self, fake_sqlalchemy_engine, embeddings_h5_path, granularity):
        model = 'SBERT'
        query_text = 'I want to know everything about the universe.'
        k = 3

        emb_mod = Mock()
        emb_mod.preprocess.return_value = query_text
        emb_mod.embed.return_value = np.ones((2,))

        # only take populated rows
        indices = H5.find_populated_rows(embeddings_h5_path, model)
        precomputed_embeddings = H5.load(embeddings_h5_path, model, indices=indices)
        precomputed_embeddings = torch.from_numpy(precomputed_embeddings)
        embedding_models = {model: emb_mod}
        norm = torch.norm(input=precomputed_embeddings, dim=1, keepdim=True)
        norm[norm == 0] = 1
        precomputed_embeddings /= norm
        precomputed_embeddings = {model: precomputed_embeddings}
        search_engine = SearchEngine(embedding_models,
                                     precomputed_embeddings,
                                     indices,
                                     fake_sqlalchemy_engine)

        deprioritized_text = 'Vegetables'
        deprioritized_strength = 'Mild'

        exclusion_text = """dshdasihsdakjbsadhbasoiasdhkjbad \n
                         hasdn,asdnioahsdpihapsdipas;kdn \n
                         jpaijdspojasdn opjpo"""

        top_indices, similarities, stats = search_engine.query(which_model=model,
                                                               granularity=granularity,
                                                               query_text=query_text,
                                                               deprioritize_text=None,
                                                               deprioritize_strength=deprioritized_strength,
                                                               has_journal=True,
                                                               is_english=True,
                                                               date_range=(2000, 2021),
                                                               exclusion_text=exclusion_text,
                                                               k=k)
        assert isinstance(stats, dict)
        assert emb_mod.preprocess.call_count == 1
        assert emb_mod.embed.call_count == 1

        if granularity == 'sentences':
            assert top_indices.shape == (k,)
            assert similarities.shape == (k,)
        elif granularity == 'articles':
            sentences_ids = ', '.join(str(id_) for id_ in top_indices)
            articles_id = fake_sqlalchemy_engine.execute(f"""SELECT DISTINCT(article_id) FROM sentences
                                                             WHERE sentence_id IN
                                                             ({sentences_ids})""").fetchall()
            assert len(articles_id) == k

        top_indices, similarities, stats = search_engine.query(which_model=model,
                                                               granularity=granularity,
                                                               query_text=query_text,
                                                               deprioritize_text=deprioritized_text,
                                                               date_range=(3000, 3001),
                                                               k=k)

        assert top_indices.shape == (0,)
        assert similarities.shape == (0,)
        assert isinstance(stats, dict)
        assert emb_mod.preprocess.call_count == 3
        assert emb_mod.embed.call_count == 3

        top_indices, similarities, stats = search_engine.query(which_model=model,
                                                               granularity=granularity,
                                                               query_text=query_text,
                                                               deprioritize_text=deprioritized_text,
                                                               k=k)
        assert isinstance(stats, dict)
        if granularity == 'sentences':
            assert top_indices.shape == (k,)
            assert similarities.shape == (k,)
        elif granularity == 'articles':
            sentences_ids = ', '.join(str(id_) for id_ in top_indices)
            articles_id = fake_sqlalchemy_engine.execute(f"""SELECT DISTINCT(article_id) FROM sentences
                                                             WHERE sentence_id IN
                                                             ({sentences_ids})""").fetchall()
            assert len(articles_id) == k

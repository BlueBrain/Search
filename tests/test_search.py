from unittest.mock import Mock

import numpy as np

from bbsearch.embedding_models import EmbeddingModel
from bbsearch.search import run_search


class TestSearch:

    def test_search(self, fake_db_cursor, embeddings_path):
        model = 'SBERT'
        query_text = 'I want to know everything about the universe.'
        k = 5

        emb_mod = Mock(spec=EmbeddingModel)
        emb_mod.preprocess.return_value = query_text
        emb_mod.embed.return_value = np.ones((2,))

        precomputed_embeddings = np.load(str(embeddings_path / model / f'{model}.npy'))

        deprioritized_text = 'Vegetables'
        deprioritized_strength = 'Mild'

        exclusion_text = """dshdasihsdakjbsadhbasoiasdhkjbad \n
                         hasdn,asdnioahsdpihapsdipas;kdn \n
                         jpaijdspojasdn opjpo"""

        indices, similarities, stats = run_search(embedding_model=emb_mod,
                                                  precomputed_embeddings=precomputed_embeddings,
                                                  database=fake_db_cursor,
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
                                                  database=fake_db_cursor,
                                                  query_text=query_text,
                                                  deprioritize_text=deprioritized_text,
                                                  date_range=(3000, 3001),
                                                  k=k)

        assert indices.shape == (0,)
        assert similarities.shape == (0,)
        assert isinstance(stats, dict)
        assert emb_mod.preprocess.call_count == 3
        assert emb_mod.embed.call_count == 3

        _, _, _ = run_search(embedding_model=emb_mod,
                             precomputed_embeddings=precomputed_embeddings,
                             database=fake_db_cursor,
                             query_text=query_text,
                             deprioritize_text=deprioritized_text,
                             k=k)

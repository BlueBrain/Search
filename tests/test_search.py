from unittest.mock import Mock

import numpy as np

from bbsearch.embedding_models import EmbeddingModel
from bbsearch.search import search


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

        indices, similarities, stats = search(embedding_model=emb_mod,
                                              precomputed_embeddings=precomputed_embeddings,
                                              database=fake_db_cursor,
                                              query_text=query_text,
                                              deprioritize_text=deprioritized_text,
                                              deprioritize_strength=deprioritized_strength,
                                              k=k)

        assert indices.shape == (k,)
        assert similarities.shape == (k,)
        assert isinstance(stats, dict)


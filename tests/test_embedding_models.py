import os
from pathlib import Path
import pytest

import numpy as np

from bbsearch.embedding_models import EmbeddingModels
from bbsearch.sql import DatabaseCreation

VERSION = 'test'


@pytest.mark.slow
class TestEmbeddingsModels:
    """Tests the Sentences Embeddings Models."""

    @classmethod
    def setup_class(cls):
        """Creation of a database and loads all the Embeddings Models. """
        cls.database_path = Path(f'cord19_{VERSION}.db')
        os.remove(str(cls.database_path))
        db = DatabaseCreation(data_path=Path('tests/data/'),
                              cord_path=Path('tests/data/CORD19_samples/'),
                              version=VERSION)
        db.construct()
        # BSV and SBERT are currently not kept in the models_to_load for test purposes
        print('Loading models ...')
        cls.models = ["USE", "SBIOBERT"]
        cls.EmbeddingModels = EmbeddingModels(assets_path='tests/assets/',
                                              models_to_load=cls.models)
        print('Models loaded')

        cls.Embeddings_path = list()
        for model_name in cls.models:
            path1 = f"{model_name}_sentence_embeddings_merged_synonyms.npz"
            path2 = f"{model_name}_sentence_embeddings.npz"
            cls.Embeddings_path.extend([path1, path2])

    @classmethod
    def teardown_class(cls):
        """Removes the database and the embeddings files created."""
        os.remove(str(cls.database_path))
        for path in cls.Embeddings_path:
            try:
                os.remove(path)
            except FileNotFoundError:
                print(f"{path} was not founded and thus not deleted")

    def test_embed_sentences(self):
        """Tests Sentences Embeddings are computed and stored in a numpy array."""
        sents = ['This is a test sentence', 'This is a second test sentence']
        for model_name in self.models:
            emdeddings = self.EmbeddingModels.embed_sentences(sents, model_name)
            assert isinstance(emdeddings, np.ndarray)
            assert emdeddings.shape[0] == len(sents)

    def test_compute_embed_sentences(self):
        """Tests Sentences Embeddings are computed for a given database."""
        for model_name in self.models:
            all_embeddings_and_ids = self.EmbeddingModels.compute_sentences_embeddings(database_path=self.database_path,
                                                                                       model_name=model_name)
            assert isinstance(all_embeddings_and_ids, dict)
            assert model_name in all_embeddings_and_ids.keys()
            assert isinstance(all_embeddings_and_ids[model_name], np.ndarray)

    def test_saving_emb_sentences(self):
        """Tests that Sentences Embeddings are saved in corresponding files."""
        self.EmbeddingModels.save_sentence_embeddings(database_path=self.database_path,
                                                      synonym_merging=False)
        self.EmbeddingModels.save_sentence_embeddings(database_path=self.database_path,
                                                      synonym_merging=True)
        for path in self.Embeddings_path:
            assert Path(path).exists()

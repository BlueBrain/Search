import nltk
import numpy as np
import sqlite3

from .searcher import Searcher
from .search import run_search
from .embedding_models import BSV, SBioBERT


class LocalSearcher(Searcher):

    def __init__(self, trained_models_path, embeddings_path, databases_path):

        self.embedding_models = {
            "BSV":  BSV(checkpoint_model_path=trained_models_path / 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'),
            "SBioBERT": SBioBERT()
        }
        self.precomputed_embeddings = {
            model_name: np.load(embeddings_path / f'{model_name}.npy').astype('float32')
            for model_name in self.embedding_models}
        # astype('float32') speeds up the search

        db_path = databases_path / "cord19.db"
        assert db_path.is_file()
        self.database_connection = sqlite3.connect(str(db_path))

    def query(self,
              which_model,
              k,
              query_text,
              has_journal=False,
              date_range=None,
              deprioritize_strength='None',
              exclusion_text=None,
              deprioritize_text=None,
              verbose=True):

        results = run_search(
            self.embedding_models[which_model],
            self.precomputed_embeddings[which_model],
            self.database_connection.cursor(),
            k,
            query_text,
            has_journal,
            date_range,
            deprioritize_strength,
            exclusion_text,
            deprioritize_text,
            verbose)

        return results

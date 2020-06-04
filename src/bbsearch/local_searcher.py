import logging
import pathlib

import numpy as np
import sqlite3

from .searcher import Searcher
from .search import run_search
from .embedding_models import BSV, SBioBERT


logger = logging.getLogger(__name__)


class LocalSearcher(Searcher):

    def __init__(self, trained_models_path, embeddings_path, databases_path):
        self.trained_models_path = pathlib.Path(trained_models_path)
        self.embeddings_path = pathlib.Path(embeddings_path)
        self.databases_path = pathlib.Path(databases_path)

        logger.info("Initializing embedding models...")
        bsv_model_name = "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        bsv_model_path = self.trained_models_path / bsv_model_name
        self.embedding_models = {
            "BSV":  BSV(checkpoint_model_path=bsv_model_path),
            "SBioBERT": SBioBERT()
        }

        logger.info("Loading precomputed embeddings...")
        self.precomputed_embeddings = {
            model_name: np.load(self.embeddings_path / f"{model_name}.npy").astype(np.float32)
            for model_name in self.embedding_models}
        # astype(np.float32) speeds up the search

        logger.info("Connecting to the Cord19 database...")
        db_path = self.databases_path / "cord19.db"
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

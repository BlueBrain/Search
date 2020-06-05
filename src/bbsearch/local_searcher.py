"""Local BBS Searcher."""

import logging
import pathlib

import numpy as np
import sqlite3

from .search import run_search
from .embedding_models import BSV, SBioBERT


logger = logging.getLogger(__name__)


class LocalSearcher:
    """Search locally using assets on disk.

    This class requires for several deep-learning modules
    to be loaded and for pre-trained models, pre-computed
    embeddings, and the SQL database to be loaded in memory.

    This is more or less a wrapper around `run_search`
    from `bbsearch.search`.

    Parameters
    ----------
    trained_models_path : str or pathlib.Path
        The folder containing pre-trained models.
    embeddings_path : str or pathlib.Path
        The folder containing pre-computed embeddings.
    databases_path : str or pathlib.Path
        The folder containing the SQL databases.
    """

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
        self.database_path = self.databases_path / "cord19.db"
        assert self.database_path.is_file()

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
        """Do the search.

        Parameters
        ----------
        which_model : str
            The name of the model to use.
        k : int
            Number of top results to display.
        query_text : str
            Query.
        has_journal : bool
            If True, only consider papers that have a journal information.
        date_range : tuple
            Tuple of form (start_year, end_year) representing the considered
            time range.
        deprioritize_text : str
            Text query of text to be deprioritized.
        deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
            How strong the deprioritization is.
        exclusion_text : str
            New line separated collection of strings that are automatically
            used to exclude a given sentence.
        verbose : bool
            If True, then printing statistics to standard output.

        Returns
        -------
        results : tuple
            All results returned by `run_search`.
        """
        with sqlite3.connect(str(self.database_path)) as database_connection:
            results = run_search(
                self.embedding_models[which_model],
                self.precomputed_embeddings[which_model],
                database_connection.cursor(),
                k,
                query_text,
                has_journal,
                date_range,
                deprioritize_strength,
                exclusion_text,
                deprioritize_text,
                verbose)

        return results

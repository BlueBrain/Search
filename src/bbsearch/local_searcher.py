"""Local BBS Searcher."""

import logging
import pathlib

import sqlite3

from .search import run_search


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
    embedding_models : dict
        The pre-trained models.
    precomputed_embeddings : dict
        The pre-computed embeddings.
    databases_path : str or pathlib.Path
        The folder containing the SQL databases.
    """

    def __init__(self, embedding_models, precomputed_embeddings, databases_path):
        self.embedding_models = embedding_models
        self.precomputed_embeddings = precomputed_embeddings
        self.databases_path = pathlib.Path(databases_path)

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

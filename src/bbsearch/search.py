"""Collection of functions focused on searching."""
import logging

import numpy as np
import torch
import torch.nn.functional as nnf

from .sql import SentenceFilter
from .utils import Timer

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
    indices : np.ndarray
        1D array containing sentence_ids corresponding to the rows of each of the
        values of precomputed_embeddings.
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        The database connection.
    """

    def __init__(self, embedding_models, precomputed_embeddings, indices, connection):
        self.embedding_models = embedding_models
        self.precomputed_embeddings = precomputed_embeddings
        self.indices = indices
        self.connection = connection

    def query(self,
              which_model,
              k,
              query_text,
              has_journal=False,
              date_range=None,
              deprioritize_strength='None',
              exclusion_text="",
              inclusion_text="",
              deprioritize_text=None,
              verbose=True,
              ):
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
            New line separated collection of strings that are automatically used to exclude a given
            sentence. If a sentence contains any of these strings then we filter it out.
        inclusion_text : str
            New line separated collection of strings. Only sentences that contain all of these
            strings are going to make it through the filtering.
        verbose : bool
            If True, then printing statistics to standard output.

        Returns
        -------
        results : tuple
            All results returned by `run_search`.
        """
        results = run_search(
            self.embedding_models[which_model],
            self.precomputed_embeddings[which_model],
            self.indices,
            self.connection,
            k,
            query_text,
            has_journal,
            date_range,
            deprioritize_strength,
            exclusion_text,
            inclusion_text,
            deprioritize_text,
            verbose)

        return results


def run_search(
        embedding_model,
        precomputed_embeddings,
        indices,
        connection,
        k,
        query_text,
        has_journal=False,
        date_range=None,
        deprioritize_strength='None',
        exclusion_text="",
        inclusion_text="",
        deprioritize_text=None,
        verbose=True
):
    """Generate search results.

    Parameters
    ----------
    embedding_model : bbsearch.embedding_models.EmbeddingModel
        Instance of EmbeddingModel of the model we want to use.

    precomputed_embeddings : torch.Tensor
        2D array containing embeddings of the model corresponding of embedding_model. Rows are
        sentences and columns are different dimensions. The embeddings need to be normalized and
        dtype float32.

    indices : np.ndarray
        1D array containing sentence_ids corresponding to the rows of `precomputed_embeddings`.

    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database.

    k : int
        Number of top results to display.

    query_text : str
        Query.

    has_journal : bool
        If True, only consider papers that have a journal information.

    date_range : tuple
        Tuple of form (start_year, end_year) representing the considered time range.

    deprioritize_text : str
        Text query of text to be deprioritized.

    deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
        How strong the deprioritization is.

    exclusion_text : str
        New line separated collection of strings that are automatically used to exclude a given sentence.
        If a sentence contains any of these strings then we filter it out.

    inclusion_text : str
        New line separated collection of strings. Only sentences that contain all of these
        strings are going to make it through the filtering.

    verbose : bool
        If True, then printing statistics to standard output.

    Returns
    -------
    sentence_ids : np.array
        1D array representing the indices of the top `k` most relevant sentences.
        The size of this array is going to be either (k, ) or (len(restricted_sentences_ids), ).

    similarities : np.array
        1D array reresenting the similarities for each of the top `k` sentences. Note that this will
        include the deprioritization part.

    stats : dict
        Various statistics. There are following keys:
        - 'query_embed_time' - how much time it took to embed the `query_text` in seconds
        - 'deprioritize_embed_time' - how much time it took to embed the `deprioritize_text` in seconds
        -
    """
    logger.info("Starting run_search")

    # Replace empty `deprioritize_text` by None
    if deprioritize_text is not None and len(deprioritize_text.strip()) == 0:
        deprioritize_text = None

    timer = Timer(verbose=verbose)

    with timer('query_embed'):
        logger.info("Embedding the query text")
        preprocessed_query_text = embedding_model.preprocess(query_text)
        embedding_query = embedding_model.embed(preprocessed_query_text)
        embedding_query = torch.from_numpy(embedding_query).to(dtype=torch.float32)

    if deprioritize_text is None:
        combined_embeddings = embedding_query
    else:
        with timer('deprioritize_embed'):
            logger.info("Embedding the deprioritization text")
            preprocessed_deprioritize_text = embedding_model.preprocess(deprioritize_text)
            embedding_deprioritize = embedding_model.embed(preprocessed_deprioritize_text)
            embedding_deprioritize = torch.from_numpy(embedding_deprioritize).to(dtype=torch.float32)

        deprioritizations = {
            'None': (1, 0),
            'Weak': (0.9, 0.1),
            'Mild': (0.8, 0.3),
            'Strong': (0.5, 0.5),
            'Stronger': (0.5, 0.7),
        }

        logger.info("Combining embeddings")
        alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
        combined_embeddings = alpha_1 * embedding_query - alpha_2 * embedding_deprioritize

    norm = torch.norm(input=combined_embeddings).item()
    if norm == 0:
        norm = 1
    combined_embeddings /= norm

    with timer('sentences_filtering'):
        logger.info("Applying sentence filtering")
        restricted_sentence_ids = torch.from_numpy((
            SentenceFilter(connection)
            .only_with_journal(has_journal)
            .date_range(date_range)
            .exclude_strings(exclusion_text.split('\n'))
            .include_strings(inclusion_text.split('\n'))
            .run()
        ))

    if len(restricted_sentence_ids) == 0:
        logger.info("No indices left after sentence filtering. Returning.")
        return np.array([]), np.array([]), timer.stats

    # Compute similarities
    with timer('query_similarity'):
        logger.info("Computing cosine similarities for the combined query")
        similarities = nnf.linear(input=combined_embeddings,
                                  weight=precomputed_embeddings)

    logger.info("Truncating similarities to the restricted indices")
    # restricted_sentence_id=  [1, 4, 5]
    # restricted_indices = [0, 3, 4]
    # similarities = [20, 21, 22, 23, 24, 25, 26]
    # restricted_similarities = [20, 23, 24]
    restricted_indices = restricted_sentence_ids - 1
    restricted_similarities = similarities[restricted_indices]

    logger.info(f"Sorting the similarities and getting the top {k} results")
    top_similarities, top_indices = torch.topk(restricted_similarities,
                                               min(k, len(restricted_similarities)),
                                               largest=True, sorted=True)
    # top similarities = [24, 23, 20]
    # top indices = [2, 1, 0]
    # restricted_indices[top_indices] = [4, 3, 0]
    top_sentence_ids = restricted_sentence_ids[top_indices]

    return top_sentence_ids.numpy(), top_similarities.numpy(), timer.stats

"""Collection of functions focused on searching."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .sql import ArticleConditioner, SentenceConditioner, get_ids_by_condition
from .utils import Timer


def search(embedding_model, precomputed_embeddings, database, k, query_text, has_journal=None,
           date_range=None, deprioritize_strength='None', exclusion_text=None, deprioritize_text=None, verbose=True):
    """Generate search results.

    Returns
    -------
    embedding_model : bbsearch.embedding_models.EmbeddingModel
        Instance of EmbeddingModel of the model we want to use.

    precomputed_embeddings : np.ndarray
        Embeddings of the model corresponding of embedding_model.

    database : sqlite3.Cursor
        Cursor to the database.

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

    verbose : bool
        If True, then printing statistics to standard output.

    Returns
    -------
    indices : np.array
        1D array representing the indices of the top `k` most relevant sentences.

    similarities : np.array
        1D array reresenting the similarities for each of the top `k` sentences. Note that this will
        include the deprioritization part.

    stats : dict
        Various statistics. There are following keys:
        - 'query_embed_time' - how much time it took to embed the `query_text` in seconds
        - 'deprioritize_embed_time' - how much time it took to embed the `deprioritize_text` in seconds
        -
    """
    timer = Timer(verbose=verbose)

    with timer('query_embed'):
        preprocessed_query_text = embedding_model.preprocess(query_text)
        embedding_query = embedding_model.embed(preprocessed_query_text)

    if deprioritize_text is not None:
        with timer('deprioritize_embed'):
            preprocessed_deprioritize_text = embedding_model.preprocess(deprioritize_text)
            embedding_deprioritize = embedding_model.embed(preprocessed_deprioritize_text)

    with timer('sentences_conditioning'):
        restricted_sentence_ids = get_ids_by_condition([], 'sentences', database)

    # Apply date-range and has-journal filtering to arr
    idx_col = precomputed_embeddings[:, 0]

    with timer('considered_embeddings_lookup'):
        mask = np.isin(idx_col, restricted_sentence_ids)

    precomputed_embeddings = precomputed_embeddings[mask]
    if len(precomputed_embeddings) == 0:
        return np.array([]), np.array([]), timer.stats

    # Compute similarities
    sentence_ids, embeddings_corpus = precomputed_embeddings[:, 0], precomputed_embeddings[:, 1:]
    with timer('query_similarity'):
        similarities_query = cosine_similarity(X=embedding_query[None, :],
                                               Y=embeddings_corpus).squeeze()

    if deprioritize_text is not None:
        with timer('deprioritize_similarity'):
            similarities_deprio = cosine_similarity(X=embedding_deprioritize[None, :],
                                                    Y=embeddings_corpus).squeeze()
    else:
        similarities_deprio = np.zeros_like(similarities_query)

    deprioritizations = {
        'None': (1, 0),
        'Weak': (0.9, 0.1),
        'Mild': (0.8, 0.3),
        'Strong': (0.5, 0.5),
        'Stronger': (0.5, 0.7),
    }
    # now: maximize L = a1 * cos(x, query) - a2 * cos(x, exclusions)
    alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
    similarities = alpha_1 * similarities_query - alpha_2 * similarities_deprio

    with timer('sorting'):
        indices = np.argsort(-similarities)[:k]

    return indices, similarities[indices], timer.stats

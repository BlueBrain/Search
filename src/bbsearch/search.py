"""Collection of functions focused on searching."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .sql import ArticleConditioner, SentenceConditioner, get_ids_by_condition
from .utils import Timer


def search(embedding_models, precomputed_embeddings, all_data, model, k, query_text, has_journal,
           date_range, deprioritize_text, deprioritize_strength, exclusion_text, verbose=True):
    """Generate search results.

    Returns
    -------
    embeding_models : bbsearch.embedding_models.EmbeddingModels
        Instance of the ``EmbeddingModels``.

    precomputed_embeddings : bbssearch.precomputed_embeddings.PrecomputedEmeddings
        Instance of the `PrecomputedEmeddings`.

    all_data : bbearch.data.AllData
        Instance of the ``AllData``.

    model : str
        Which model to use for embeddings.

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

    query_text = [query_text]
    deprioritize_text = [deprioritize_text]

    with timer('query_embed'):
        preprocessed_query_text = embedding_models.preprocess(query_text, model)
        embedding_query = embedding_models.embed(preprocessed_query_text, model)

    if deprioritize_text[0]:
        with timer('deprioritize_embed'):
            preprocessed_deprioritize_text = embedding_models.preprocess(deprioritize_text, model)
            embedding_deprioritize = embedding_models.embed(preprocessed_deprioritize_text, model)

    # Apply article conditions
    article_conditions = [ArticleConditioner.get_date_range_condition(date_range)]
    if has_journal:
        article_conditions.append(ArticleConditioner.get_has_journal_condition())

    with timer('article_conditioning'):
        restricted_article_ids = get_ids_by_condition(article_conditions, 'articles', all_data.db)

    # Apply sentence conditions
    all_aticle_ids_str = ', '.join([f"'{sha}'" for sha in restricted_article_ids])
    sentence_conditions = [f"Article IN ({all_aticle_ids_str})",
                           SentenceConditioner.get_restrict_to_tag_condition("COVID-19")]

    excluded_words = [x for x in exclusion_text.lower().split('\n') if x]  # remove empty strings
    sentence_conditions += [SentenceConditioner.get_word_exclusion_condition(word) for word in excluded_words]

    with timer('sentences_conditioning'):
        restricted_sentence_ids = get_ids_by_condition(sentence_conditions, 'sections', all_data.db)

    # Load sentence embedding from the npz file
    with timer('embeddings_load'):

            arr = all_models.embeddings[model]

    # Apply date-range and has-journal filtering to arr
    idx_col = arr[:, 0]

    with timer('considered_embeddings_lookup'):
        mask = np.isin(idx_col, restricted_sentence_ids)

    arr = arr[mask]
    if len(arr) == 0:
        return np.array([]), np.array([]), timer.stats

    # Compute similarities
    sentence_ids, embeddings_corpus = arr[:, 0], arr[:, 1:]
    with timer('query_similarity'):
        similarities_query = cosine_similarity(X=embedding_query,
                                               Y=embeddings_corpus).squeeze()

    if deprioritize_text[0]:
        with timer('deprioritize_similarity'):
            similarities_exclu = cosine_similarity(X=embedding_exclu,
                                                   Y=embeddings_corpus).squeeze()
    else:
        similarities_exclu = np.zeros_like(similarities_query)

    deprioritizations = {
        'None': (1, 0),
        'Weak': (0.9, 0.1),
        'Mild': (0.8, 0.3),
        'Strong': (0.5, 0.5),
        'Stronger': (0.5, 0.7),
    }
    # now: maximize L = a1 * cos(x, query) - a2 * cos(x, exclusions)
    alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
    similarities = alpha_1 * similarities_query - alpha_2 * similarities_exclu

    with timer('sorting'):
        indices = np.argsort(-similarities)[:k]

    return indices, similarities[indices], timer.stats

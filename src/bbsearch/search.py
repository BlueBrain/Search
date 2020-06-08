"""Collection of functions focused on searching."""
import pathlib
import sqlite3

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .sql import ArticleConditioner, SentenceConditioner, get_ids_by_condition, get_shas_from_ids
from .utils import Timer


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
        if not self.database_path.is_file():
            raise FileNotFoundError('{} does not exist'.format(self.database_path))

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


def filter_sentences(database,
                     has_journal=False,
                     date_range=None,
                     exclusion_text=None):
    """Filter sentences based on specified conditions.

    Parameters
    ----------
    database : sqlite3.Cursor
        Cursor to the database.

    has_journal : bool
        If True, only consider papers that have a journal information.

    date_range : tuple
        Tuple of form (start_year, end_year) representing the considered time range.

    exclusion_text : str
        New line separated collection of strings that are automatically used to exclude a given sentence.

    Returns
    -------
    restricted_sentence_ids: list
        List of the sentences ids after the filtration related to the criteria specified by the user.
    """
    # Apply article conditions
    article_conditions = []
    if date_range is not None:
        article_conditions.append(ArticleConditioner.get_date_range_condition(date_range))
    if has_journal:
        article_conditions.append(ArticleConditioner.get_has_journal_condition())
    article_conditions.append(ArticleConditioner.get_restrict_to_tag_condition('has_covid19_tag'))

    restricted_article_ids = get_ids_by_condition(article_conditions, 'articles', database)

    # Articles ID to SHA
    all_article_shas_str = ', '.join([f"'{sha}'"
                                      for sha in get_shas_from_ids(restricted_article_ids, database)])
    sentence_conditions = [f"sha IN ({all_article_shas_str})"]

    # Apply sentence conditions
    if exclusion_text is not None:
        excluded_words = filter(lambda word: len(word) > 0, exclusion_text.lower().split('\n'))
        sentence_conditions += [SentenceConditioner.get_word_exclusion_condition(word)
                                for word in excluded_words]
    restricted_sentence_ids = get_ids_by_condition(sentence_conditions, 'sentences', database)

    return restricted_sentence_ids


def run_search(embedding_model, precomputed_embeddings, database, k, query_text, has_journal=False, date_range=None,
               deprioritize_strength='None', exclusion_text=None, deprioritize_text=None, verbose=True):
    """Generate search results.

    Parameters
    ----------
    embedding_model : bbsearch.embedding_models.EmbeddingModel
        Instance of EmbeddingModel of the model we want to use.

    precomputed_embeddings : np.ndarray
        Embeddings of the model corresponding of embedding_model.
        The first column has to be the corresponding index of the sentence in the database.
        The others columns have to be the embeddings, so need to have the same size as the model specified requires.

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
    sentence_ids : np.array
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
        restricted_sentence_ids = filter_sentences(database,
                                                   has_journal=has_journal,
                                                   date_range=date_range,
                                                   exclusion_text=exclusion_text)

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

    return sentence_ids[indices], similarities[indices], timer.stats

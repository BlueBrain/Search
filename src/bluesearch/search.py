"""Collection of functions focused on searching."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import logging

import numpy as np
import torch
import torch.nn.functional as nnf

from .sql import SentenceFilter, retrieve_article_ids
from .utils import Timer

logger = logging.getLogger(__name__)


class SearchEngine:
    """Search locally using assets on disk.

    This class requires for several deep-learning modules
    to be loaded and for pre-trained models, pre-computed
    embeddings, and the SQL database to be loaded in memory.

    This is more or less a wrapper around `run_search`
    from `bluesearch.search`.

    Parameters
    ----------
    embedding_models : dict
        The pre-trained models.
    precomputed_embeddings : dict
        The pre-computed embeddings.
    indices : np.ndarray
        1D array containing sentence_ids corresponding to the rows of each of the
        values of precomputed_embeddings.
    connection : sqlalchemy.engine.Engine
        The database connection.
    """

    def __init__(self, embedding_models, precomputed_embeddings, indices, connection):
        self.embedding_models = embedding_models
        self.precomputed_embeddings = precomputed_embeddings
        self.indices = indices
        self.connection = connection
        logger.info("Retrieving articles ids for all sentence ids...")
        self.all_article_ids = retrieve_article_ids(self.connection)
        logger.info("Retrieve articles ids: DONE")

    def query(
        self,
        which_model,
        k,
        query_text,
        granularity="sentences",
        has_journal=False,
        is_english=True,
        discard_bad_sentences=False,
        date_range=None,
        deprioritize_strength="None",
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
        granularity : str
            One of ('sentences', 'articles'). Search granularity.
        has_journal : bool
            If True, only consider papers that have a journal information.
        is_english : bool
            If True, only consider papers that are in English.
        discard_bad_sentences : bool
            If True, then all sentences with marked as bad quality will be
            discarded.
        date_range : tuple
            Tuple of form (start_year, end_year) representing the considered
            time range.
        deprioritize_text : str
            Text query of text to be deprioritized.
        deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
            How strong the deprioritization is.
        exclusion_text : str
            New line separated collection of strings that are automatically
            used to exclude a given sentence. If a sentence contains any of
            these strings then we filter it out.
        inclusion_text : str
            New line separated collection of strings. Only sentences that
            contain all of these strings are going to make it through the
            filtering.
        verbose : bool
            If True, then printing statistics to standard output.

        Returns
        -------
        sentence_ids : np.array
            1D array representing the indices of the top `k` most relevant
            sentences. The size of this array is going to be either (k, ) or
            (len(restricted_sentences_ids), ).
        similarities : np.array
            1D array reresenting the similarities for each of the top `k`
            sentences. Note that this will include the deprioritization part.
        stats : dict
            Various statistics. There are following keys:

            - 'query_embed_time' - how much time it took to embed the
              `query_text` in seconds
            - 'deprioritize_embed_time' - how much time it took to embed the
              `deprioritize_text` in seconds
        """
        embedding_model = self.embedding_models[which_model]
        precomputed_embeddings = self.precomputed_embeddings[which_model]

        logger.info("Starting run_search")

        # Replace empty `deprioritize_text` by None
        if deprioritize_text is not None and len(deprioritize_text.strip()) == 0:
            deprioritize_text = None

        timer = Timer(verbose=verbose)

        with timer("query_embed"):
            logger.info("Embedding the query text")
            preprocessed_query_text = embedding_model.preprocess(query_text)
            embedding_query = embedding_model.embed(preprocessed_query_text)
            embedding_query = torch.from_numpy(embedding_query).to(dtype=torch.float32)

        if deprioritize_text is None:
            combined_embeddings = embedding_query
        else:
            with timer("deprioritize_embed"):
                logger.info("Embedding the deprioritization text")
                preprocessed_deprioritize_text = embedding_model.preprocess(
                    deprioritize_text
                )
                embedding_deprioritize = embedding_model.embed(
                    preprocessed_deprioritize_text
                )
                embedding_deprioritize = torch.from_numpy(embedding_deprioritize).to(
                    dtype=torch.float32
                )

            deprioritizations = {
                "None": (1, 0),
                "Weak": (0.9, 0.1),
                "Mild": (0.8, 0.3),
                "Strong": (0.5, 0.5),
                "Stronger": (0.5, 0.7),
            }

            logger.info("Combining embeddings")
            alpha_1, alpha_2 = deprioritizations[deprioritize_strength]
            combined_embeddings = (
                alpha_1 * embedding_query - alpha_2 * embedding_deprioritize
            )

        norm = torch.norm(input=combined_embeddings).item()
        if norm == 0:
            norm = 1
        combined_embeddings /= norm

        with timer("sentences_filtering"):
            logger.info("Applying sentence filtering")
            restricted_sentence_ids = torch.from_numpy(
                (
                    SentenceFilter(self.connection)
                    .only_english(is_english)
                    .only_with_journal(has_journal)
                    .discard_bad_sentences(discard_bad_sentences)
                    .date_range(date_range)
                    .exclude_strings(exclusion_text.split("\n"))
                    .include_strings(inclusion_text.split("\n"))
                    .run()
                )
            )

        if len(restricted_sentence_ids) == 0:
            logger.info("No indices left after sentence filtering. Returning.")
            return np.array([]), np.array([]), timer.stats

        # Compute similarities
        with timer("query_similarity"):
            logger.info("Computing cosine similarities for the combined query")
            similarities = nnf.linear(
                input=combined_embeddings, weight=precomputed_embeddings
            )

        logger.info(f"Sorting the similarities and getting the top {k} results")
        top_sentence_ids, top_similarities = self.get_top_k_results(
            k, similarities, restricted_sentence_ids, granularity=granularity
        )

        return top_sentence_ids.numpy(), top_similarities.numpy(), timer.stats

    def get_top_k_results(
        self, k, similarities, restricted_sentence_ids, granularity="sentences"
    ):
        """Retrieve top k results (granularity sentences or articles).

        Parameters
        ----------
        k : int
            Top k results to retrieve.
        similarities : torch.Tensor
            Similarities values
        restricted_sentence_ids : torch.Tensor
            Tensor containing the sentences_ids to keep for the top k retrieving.
        granularity : str
            One of ('sentences', 'articles').

        Returns
        -------
        top_sentence_ids : torch.Tensor
            1D array representing the indices of the top `k` most relevant
            sentences. The size of this array is going to be either (k, ) or
            (len(restricted_sentences_ids), ). k being equal to k for
            granularity = 'sentences', and num of sentences for k unique
            articles for granularity = 'articles'.
        top_similarities : torch.Tensor
            1D array representing the similarities for each of the top `k` sentences.
        """
        logger.info("Truncating similarities to the restricted indices")
        # restricted_sentence_id=  [1, 4, 5]
        # restricted_indices = [0, 3, 4]
        # similarities = [20, 21, 22, 23, 24, 25, 26]
        # restricted_similarities = [20, 23, 24]
        restricted_indices = restricted_sentence_ids - 1
        restricted_similarities = similarities[restricted_indices]

        if granularity == "sentences":
            logger.info(
                f"Sorting the similarities and getting the top {k} sentences results"
            )
            top_similarities, top_indices = torch.topk(
                restricted_similarities,
                min(k, len(restricted_similarities)),
                largest=True,
                sorted=True,
            )
            top_sentence_ids = restricted_sentence_ids[top_indices]
            # top similarities = [24, 23, 20]
            # top indices = [2, 1, 0]
            # restricted_indices[top_indices] = [4, 3, 0]

        elif granularity == "articles":
            logger.info(
                f"Sorting the similarities and getting the top {k} articles results"
            )
            top_similarities, top_indices = torch.sort(
                restricted_similarities, descending=True
            )
            top_sentence_ids = restricted_sentence_ids[top_indices]
            article_ids = set()

            num = 0
            for sentence_id in top_sentence_ids:
                num += 1
                article_ids.add(self.all_article_ids[int(sentence_id)])
                if len(article_ids) == k:
                    break

            top_sentence_ids, top_similarities = (
                top_sentence_ids[:num],
                top_similarities[:num],
            )

        else:
            raise NotImplementedError(f"{granularity} not implemented ")

        return top_sentence_ids, top_similarities

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


class QueryMachine:
    """Class used for querying from the embedding database.

    Parameters
    ----------
    embedder : bbcov.Embedder

    df_docs : pandas.DataFrame
        The rows represent different documents. The columns are metadata.

    text_column_name : str
        The name of the column of `df_docs` to use for embeddings.

    metric : str or callable
        If string then it should be a valid metric parameter for
        `sklearn.metrics.pairwise_distances`

    Attributes
    ----------
    doc_embeddings : pandas.DataFrame
        Rows represent different
    """

    def __init__(self, embedder, df_docs, text_column_name, metric='cosine'):
        self.df_docs = df_docs
        self.text_column_name = text_column_name
        self.embedder = embedder
        self.metric = metric

        self._precompute_doc_embeddings()

    def _precompute_doc_embeddings(self):
        texts = self.df_docs[self.text_column_name]
        embeddings = self.embedder(texts)
        self.doc_embeddings = pd.DataFrame(embeddings, index=self.df_docs.index)

    def get_distances(self, query_embedding, doc_embeddings):
        # TODO: handle metric
        distances = pairwise_distances(
            query_embedding,
            doc_embeddings, metric=self.metric)
        return distances

    def get_top_k(self, query, embeddings, k=3, exclusion_query=None):
        """

        Parameters
        ----------
        query : str
            A single query text
        embeddings : pandas.DataFrame
            An array of shape (n_documents, d_embedding)
        df_mask : array_like or None

        exclusion_query : str or None
            Exclusion query. If None then not considered.

        Returns
        -------
        top_k : pandas.Index
            Indices of the top k matches in the dataframe
        top_k_distances : list_like
            Distance of the top k matches to the query string
        """
        alpha = 0.5

        query_embedding = self.embedder(query)

        distances = self.get_distances(
            query_embedding,
            embeddings.values)
        distances = distances[0]

        if exclusion_query is not None:
            exclusion_embedding = self.embedder(exclusion_query)

            exclusion_distances = self.get_distances(
                exclusion_embedding,
                embeddings.values)
            exclusion_distances = exclusion_distances[0]
        else:
            exclusion_distances = np.zeros_like(distances)

        overall_distances = (1 - alpha) * distances - alpha * exclusion_distances

        top_k = overall_distances.argsort()[:k]
        top_k_distances = overall_distances[top_k]

        return embeddings.index[top_k], top_k_distances

    def get_top_k_docs(self, query, k=3):
        # TODO: instead of indices return something nice
        top_k, top_k_distances = self.get_top_k(
            query,
            self.doc_embeddings,
            k=k)
        return top_k

    def get_most_relevant_paragraphs(self, query, doc_ids):
        most_relevant_paragraphs = []
        paragraph_distances = []
        for idx in doc_ids:
            text = self.df_docs.loc[idx][self.text_column_name]
            paragraphs = text.split('\n')
            embeddings = pd.DataFrame(self.embedder(paragraphs))
            top_par, top_par_dist = self.get_top_k(query, embeddings, k=1)
            most_relevant_paragraphs.append(paragraphs[top_par[0]])
            paragraph_distances.append(top_par_dist[0])

        return most_relevant_paragraphs, paragraph_distances

    def print_query(self, query, k=3, df_mask=None, exclusion_query=None):
        """

        Parameters
        ----------
        query : str
        k : int
        df_mask : pandas.Series
            A boolean-valued pandas Series used to mask the original
            documents dataframe.

        exclusion_query : str or None
            Exclusion query. If None then not considered.

        """

        if df_mask is None:
            filtered_embeddings = self.doc_embeddings
        else:
            filtered_embeddings = self.doc_embeddings[df_mask]

        top_k, top_k_distances = self.get_top_k(
            query,
            filtered_embeddings,
            k=k,
            exclusion_query=exclusion_query)

        paragraphs, paragraph_distances = self.get_most_relevant_paragraphs(query, top_k)
        for i in range(len(top_k)):
            if i > 0:
                print()
                print('/\\' * 40)
                print()
            self.format_result(top_k[i], top_k_distances[i], paragraphs[i], paragraph_distances[i])

    def format_result(self, doc_idx, doc_distance, paragraph, paragraphs_distance):
        """

        Parameters
        ----------
        doc_idx : object
            An object that can be used to index `self.df_docs`. In other words,
            `self.df_docs.loc[doc_idx]` should be a valid call.
        doc_distance : float
        paragraph : str
        paragraphs_distance : float

        Returns
        -------

        """

        column_names = {
            'title': 'title',
            'authors': 'authors',
            'date': 'publish_time',
            'journal': 'journal',
            'doi': 'doi',
        }

        entry = self.df_docs.loc[doc_idx]

        print(f"Distance : {doc_distance:.3f}")
        print(f"ID       : {entry.name}")
        print(f"Title    : {entry[column_names['title']]}")
        print(f"Authors  : {entry[column_names['authors']]}")
        print(f"Date     : {entry[column_names['date']]}")
        print(f"Journal  : {entry[column_names['journal']]}")
        print(f"DOI      : {entry[column_names['doi']]}")
        print(f" Relevant paragraph (distance: {paragraphs_distance:.3f})".center(80, '-'))
        print(paragraph)

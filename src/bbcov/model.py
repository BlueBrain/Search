import abc
import numpy as np
import joblib


class Model(abc.ABC):

    def embed(self, text):
        if isinstance(text, str):
            text = [text]
        return self._embed(text)

    @abc.abstractmethod
    def _embed(self, text):
        """Compute embedding for a set of texts

        Parameters
        ----------
        text : array_like

        Returns
        -------

        """


class DummyModel(Model):

    def __init__(self, d_embedding):
        self.d_embedding = d_embedding

    def _embed(self, text):
        embedding = np.random.randn(len(text), self.d_embedding)
        return embedding


class LDAModel(Model):

    def __init__(self, vectorizer, lda):
        self.vectorizer = vectorizer
        self.lda = lda

    def _embed(self, text):
        tf = self.vectorizer.transform(text)
        topic_distributions = self.lda.transform(tf)

        return topic_distributions

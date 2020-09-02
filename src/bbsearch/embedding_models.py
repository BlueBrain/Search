"""Model handling sentences embeddings."""
import string
from abc import ABC, abstractmethod

import numpy as np
import sent2vec
import tensorflow_hub as hub
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import AutoModelWithLMHead, AutoTokenizer

from .sql import retrieve_sentences_from_sentence_ids


class EmbeddingModel(ABC):
    """Abstract interface for the Sentences Embeddings Models."""

    @property
    @abstractmethod
    def dim(self):
        """Return dimension of the embedding."""

    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...) if needed by the model.

        This is a default implementation that perform no preprocessing. Model specific
        preprocessing can be define within children classes.

        Parameters
        ----------
        raw_sentence: str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence
            Preprocessed sentence in the format expected by the model if needed.
        """
        return raw_sentence

    def preprocess_many(self, raw_sentences):
        """Preprocess multiple sentences.

        This is a default implementation and can be overridden by children classes.

        Parameters
        ----------
        raw_sentences : list[str]
            List of str representing raw sentences that we want to embed.

        Returns
        -------
        preprocessed_sentences
            List of preprocessed sentences corresponding to `raw_sentences`.
        """
        return [self.preprocess(sentence) for sentence in raw_sentences]

    @abstractmethod
    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence: str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            One dimensional vector representing the embedding of the given sentence.
        """

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for all provided sentences.

        This is a default implementation. Children classes can implement more sophisticated
        batching schemes.

        Parameters
        ----------
        preprocessed_sentences : list[str]
            List of preprocessed sentences.

        Returns
        -------
        embeddings : np.ndarray
            2D numpy array with shape `(len(preprocessed_sentences), self.dim)`. Each row
            is an embedding of a sentence in `preprocessed_sentences`.
        """
        return np.array([self.embed(sentence) for sentence in preprocessed_sentences])


class SBioBERT(EmbeddingModel):
    """Sentence BioBERT.

    Parameters
    ----------
    device: torch.device
        Torch device.

    References
    ----------
    https://huggingface.co/gsarti/biobert-nli
    """

    def __init__(self,
                 device=None):
        self.device = device or torch.device('cpu')
        self.sbiobert_model = AutoModelWithLMHead.from_pretrained("gsarti/biobert-nli").bert.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 768

    def preprocess(self, raw_sentence):
        """Preprocess the sentence - tokenization and determining of token ids.

        Note that this method already works in batched way if we pass a list.

        Parameters
        ----------
        raw_sentence: str or list[str]
            Raw sentence to embed. One can also provide multiple sentences.

        Returns
        -------
        encoding : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids', 'token_type_ids'
            and 'attention_mask'. All of the corresponding values are going to be ``torch.Tensor``
            of shape `(n_sentences, n_tokens)`.

        """
        encoding = self.tokenizer(raw_sentence,
                                  pad_to_max_length=True,
                                  return_tensors='pt'
                                  )
        return encoding

    def preprocess_many(self, raw_sentences):
        """Preprocess multiple sentences - tokenization and determining of token ids.

        Parameters
        ----------
        raw_sentences: list[str]
            List of raw sentence to embed.

        Returns
        -------
        encodings : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids', 'token_type_ids'
            and 'attention_mask'. All of the corresponding values are going to be ``torch.Tensor``
            of shape `(n_sentences, n_tokens)`.

        """
        return self.preprocess(raw_sentences)

    def embed(self, preprocessed_sentence):
        """Compute the sentence embedding for a given sentence.

        Note that this method already works in batched way if we pass a `BatchEncoding` that
        contains batches.

        Parameters
        ----------
        preprocessed_sentence: transformers.BatchEncoding
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence of shape (768,) if only a single sample in the
            batch. Otherwise `(len(preprocessed_sentences), 768)`.
        """
        with torch.no_grad():
            embedding = self.sbiobert_model(**preprocessed_sentence.to(self.device))[1].squeeze()

        return embedding.numpy()

    def embed_many(self, preprocessed_sentences):
        """Compute the sentences embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences: transformers.BatchEncoding
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence of shape `(len(preprocessed_sentences), 768)`
        """
        return self.embed(preprocessed_sentences)


class BSV(EmbeddingModel):
    """BioSentVec.

    Parameters
    ----------
    checkpoint_model_path: pathlib.Path
        Path to the file of the stored model BSV.

    References
    ----------
    https://github.com/ncbi-nlp/BioSentVec
    """

    def __init__(self,
                 checkpoint_model_path):
        self.checkpoint_model_path = checkpoint_model_path
        if not self.checkpoint_model_path.is_file():
            raise FileNotFoundError(f'The file {self.checkpoint_model_path} was not found.')
        self.bsv_model = sent2vec.Sent2vecModel()
        self.bsv_model.load_model(str(self.checkpoint_model_path))
        self.bsv_stopwords = set(stopwords.words('english'))

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 700

    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...).

        Parameters
        ----------
        raw_sentence: str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence: str
            Preprocessed sentence.
        """
        raw_sentence = raw_sentence.replace('/', ' / ')
        raw_sentence = raw_sentence.replace('.-', ' .- ')
        raw_sentence = raw_sentence.replace('.', ' . ')
        raw_sentence = raw_sentence.replace('\'', ' \' ')
        raw_sentence = raw_sentence.lower()
        tokens = [token for token in word_tokenize(raw_sentence)
                  if token not in string.punctuation and token not in self.bsv_stopwords]
        return ' '.join(tokens)

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence: str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence of shape (700,).
        """
        return self.embed_many([preprocessed_sentence]).squeeze()

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences: list[str]
            Preprocessed sentences to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentences of shape `(len(preprocessed_sentences), 700)`.
        """
        embeddings = self.bsv_model.embed_sentences(preprocessed_sentences)
        return embeddings


class SBERT(EmbeddingModel):
    """Sentence BERT.

    References
    ----------
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self):
        self.sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 768

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence: str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the given sentence of shape (768,).
        """
        return self.embed_many([preprocessed_sentence]).squeeze()

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences: list[str]
            Preprocessed sentences to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentences of shape `(len(preprocessed_sentences), 768)`.
        """
        embeddings = np.array(self.sbert_model.encode(preprocessed_sentences))
        return embeddings


class USE(EmbeddingModel):
    """Universal Sentence Encoder.

    References
    ----------
    https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder?hl=en
    """

    def __init__(self):
        self.use_version = 5
        self.use_model = hub.load(f"https://tfhub.dev/google/universal-sentence-encoder-large/{self.use_version}")

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 512

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence: str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence of shape (512,).
        """
        return self.embed_many([preprocessed_sentence]).squeeze()

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences: list[str]
            Preprocessed sentences to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentences of shape `(len(preprocessed_sentences), 512)`.
        """
        embedding = self.use_model(preprocessed_sentences).numpy()
        return embedding


def compute_database_embeddings(connection, model, indices):
    """Compute Sentences Embeddings for a given model and a given database (articles with covid19_tag True).

    Parameters
    ----------
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the database.

    model: EmbeddingModel
        Instance of the EmbeddingModel of choice.

    indices : np.ndarray
        1D array storing the sentence_ids for which we want to perform the embedding.

    Returns
    -------
    final_embeddings: np.array
        2D numpy array with all sentences embeddings for the given models. Its shape is
        `(len(retrieved_indices), dim)`.

    retrieved_indices : np.ndarray
        1D array of sentence_ids that we managed to embed. Note that the order corresponds
        exactly to the rows in `final_embeddings`.
    """
    sentences = retrieve_sentences_from_sentence_ids(indices, connection)

    all_embeddings = list()
    all_ids = list()
    num_errors = 0

    for index, row in sentences.iterrows():
        sentence_text, sentence_id = row['text'], row['sentence_id']
        try:
            preprocessed_sentence = model.preprocess(sentence_text)
            embedding = model.embed(preprocessed_sentence)
        except IndexError:
            # This could happen when the sentence is too long for example
            num_errors += 1
            continue

        all_ids.append(sentence_id)
        all_embeddings.append(embedding)

        if index % 1000 == 0:
            print(f'Embedded {index} with {num_errors} errors')

    final_embeddings = np.array(all_embeddings)
    retrieved_indices = np.array(all_ids)

    return final_embeddings, retrieved_indices

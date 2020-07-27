"""Model handling sentences embeddings."""
from abc import ABC, abstractmethod
import string

from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import pandas as pd
import sent2vec
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

from .sql import retrieve_sentences_from_sentence_id


class EmbeddingModel(ABC):
    """Abstract interface for the Sentences Embeddings Models."""

    @property
    @abstractmethod
    def dim(self):
        """Return dimension of the embedding."""

    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...) if needed by the model.

        Parameters
        ----------
        raw_sentence: str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence:
            Preprocessed sentence in the format expected by the model if needed.
        """
        return raw_sentence

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
        """Preprocess the sentence (Tokenization, ...).

        Parameters
        ----------
        raw_sentence: str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence: torch.Tensor
            Preprocessed sentence.
        """
        # Add the special tokens.
        marked_text = "[CLS] " + raw_sentence + " [SEP]"

        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        preprocessed_sentence = torch.tensor([indexed_tokens]).to(self.device)

        return preprocessed_sentence

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence: torch.Tensor
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence of shape (768,)
        """
        segments_tensors = torch.ones_like(preprocessed_sentence)
        with torch.no_grad():
            self.sbiobert_model.eval()
            encoded_layers, test = self.sbiobert_model(preprocessed_sentence, segments_tensors)
            sentence_encoding = encoded_layers[-1].squeeze().mean(axis=0)
            embedding = sentence_encoding.detach().cpu().numpy()

        return embedding


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
        embedding = self.bsv_model.embed_sentences([preprocessed_sentence])[0]
        return embedding


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
        embedding = self.sbert_model.encode([preprocessed_sentence])[0]
        return embedding


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
        embedding = self.use_model([preprocessed_sentence]).numpy()[0]
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
        Huge numpy array with all sentences embeddings for the given models.
        Format: (sentence_id, embeddings).

    retrieved_indices : np.ndarray
        1D array of sentence_ids that we managed to embed. Note that the order corresponds
        exactly to the rows in `final_embeddings`.
    """
    sentences = retrieve_sentences_from_sentence_id(indices, connection)

    all_embeddings = list()
    all_ids = list()
    num_errors = 0

    for index, row in sentences.iterrows():
        sentence_text, sentence_id = row['text'], row['sentence_id']
        try:
            preprocessed_sentence = model.preprocess(sentence_text)
            embedding = model.embed(preprocessed_sentence)
        except IndexError:
            num_errors += 1
            continue

        all_ids.append(sentence_id)
        all_embeddings.append(embedding)
        if index % 1000 == 0:
            print(f'Embedded {index} with {num_errors} errors')

    final_embeddings = np.array(all_embeddings)
    retrieved_indices = np.array(all_ids)

    return final_embeddings, retrieved_indices

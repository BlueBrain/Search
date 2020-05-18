"""Model handling sentences embeddings."""
from abc import ABC, abstractmethod
import string

from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import sent2vec
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


class EmbeddingModel(ABC):
    """Abstract interface for the Sentences Embeddings Models."""

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


def compute_database_embeddings(database, model):
    """Compute Sentences Embeddings for a given model and a given database (articles with covid19_tag True).

    Parameters
    ----------
    database: sqlite3.Cursor
        Cursor to the database with 'sentences' table.
    model: EmbeddingModel
        Instance of the EmbeddingModel of choice.

    Returns
    -------
    final_embeddings: np.array
        Huge numpy array with all sentences embeddings for the given models.
        Format: (sentence_id, embeddings).
    """
    query = """SELECT sentence_id, text FROM sentences
            WHERE sha IN (SELECT sha FROM article_id_2_sha WHERE article_id IN
            (SELECT article_id FROM articles WHERE has_covid19_tag is True))"""
    all_embeddings = list()
    all_ids = list()
    query_execution = database.execute(query)
    query_end = False
    while not query_end:
        results = query_execution.fetchone()
        if results is not None:
            sentence_id, sentence_text = results
            preprocessed_sentence = model.preprocess(sentence_text)
            embedding = model.embed(preprocessed_sentence)
            all_ids.append(sentence_id)
            all_embeddings.append(embedding)
        else:
            query_end = True

    all_embeddings = np.array(all_embeddings)
    all_ids = np.array(all_ids).reshape((-1, 1))
    final_embeddings = np.concatenate((all_ids, all_embeddings), axis=1)

    return final_embeddings

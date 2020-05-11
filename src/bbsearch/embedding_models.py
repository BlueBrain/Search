"""Model handling sentences embeddings."""
from abc import ABC, abstractmethod
import string

from nltk.corpus import stopwords
from nltk import word_tokenize
import sent2vec
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


class EmbeddingModel(ABC):
    """Abstract interface for the Sentences Embeddings Models."""

    @abstractmethod
    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...).

        Parameters
        ----------
        raw_sentence: str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence:
            Preprocessed sentence in the format expected by the model.
        """

    @abstractmethod
    def encode(self, preprocess_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocess_sentence: str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence.
        """


class SBioBERT(EmbeddingModel):

    def __init__(self,
                 device=None):

        self.device = device or torch.device('cpu')
        self.biobert_model = AutoModelWithLMHead.from_pretrained("gsarti/biobert-nli").bert.to(self.device)
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
        preprocess_sentence = torch.tensor([indexed_tokens]).to(self.device)

        return preprocess_sentence

    def encode(self, preprocess_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocess_sentence: torch.Tensor
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence.
        """

        segments_tensors = torch.ones_like(preprocess_sentence)
        with torch.no_grad():
            self.biobert_model.eval()
            encoded_layers, test = self.biobert_model(preprocess_sentence, segments_tensors)
            sentence_encoding = encoded_layers[-1].squeeze().mean(axis=0)
            embedding = sentence_encoding.detach().cpu().numpy()

        return embedding


class BSV(EmbeddingModel):

    def __init__(self,
                 assets_path):

        self.assets_path = assets_path
        self.bsv_path = self.assets_path / 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
        self.bsv_model = sent2vec.Sent2vecModel().load_model(str(self.bsv_path))
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

    def encode(self, preprocess_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocess_sentence: torch.Tensor
            Preprocessed sentence to embed.

        Returns
        -------
        embedding: numpy.array
            Embedding of the specified sentence.
        """
        embedding = self.bsv_model(preprocess_sentence)
        return embedding

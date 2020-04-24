import string
import logging
import pathlib

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import tensorflow_hub as hub
import sent2vec
from sentence_transformers import SentenceTransformer

from .s_bio_bert import SBioBERT

logger = logging.getLogger(__name__)


class EmbeddingModels:

    def __init__(self, assets_path):
        self.assets_path = pathlib.Path(assets_path)

        logger.info("Downloading NLTK modules...")
        nltk.download('punkt')
        nltk.download('stopwords')

        logger.info("Loading USE...")
        self.use_version = 5
        self.use_url = "https://tfhub.dev/google/universal-sentence-encoder-large"
        self.use = hub.load(f"{self.use_url}/{self.use_version}")

        logger.info("Loading SBERT...")
        self.sbert = SentenceTransformer('bert-base-nli-mean-tokens')

        logger.info("Loading BioSentVec...")
        self.bsv = sent2vec.Sent2vecModel()
        self.bsv.load_model(str(self.assets_path / 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'))
        self.bsv_stopwords = set(stopwords.words('english'))

        logger.info("Loading SBioBERT...")
        self.sbiobert = SBioBERT()

    def bsv_preprocess(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens = [token for token in word_tokenize(text)
                  if token not in string.punctuation and token not in self.bsv_stopwords]
        return ' '.join(tokens)

    def embed_sentences(self, sentences, embedding_name):
        embedding_name = embedding_name.upper()

        if embedding_name == 'USE':
            return self.use(sentences).numpy()
        elif embedding_name == 'SBERT':
            return np.stack(self.sbert.encode(sentences), axis=0)
        elif embedding_name == 'SBIOBERT':
            return np.stack(self.sbiobert.encode(sentences), axis=0)
        elif embedding_name == 'BSV':
            preprocessed = [self.bsv_preprocess(x) for x in sentences]
            return self.bsv.embed_sentences(preprocessed)
        else:
            raise NotImplementedError(
                f"Embedding {embedding_name} not available!")

"""The `EmbeddingModels` class."""

import string
import logging
import pathlib
import warnings

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import tensorflow_hub as hub
import sent2vec
from sentence_transformers import SentenceTransformer

from .s_bio_bert import SBioBERT

logger = logging.getLogger(__name__)


class EmbeddingModels:
    """A class for sentence embeddings.

    It holds a number of different sentence embedding models,
    and provides methods for embedding sentences using any of
    these models.
    """

    def __init__(self, assets_path, models_to_load=None):
        """Initialize class.

        Parameters
        ----------
        assets_path : str or pathlib.Path
            Path for loading serialized pre-trained models. Currently
            only used to load the BSV model.
        models_to_load : list
            A list with model names to load.
        """
        self.assets_path = pathlib.Path(assets_path)
        self.all_supported_models = ["USE", "SBERT", "BSV", "SBIOBERT"]
        self.models = dict()
        self.bsv_stopwords = set(stopwords.words('english'))

        models_to_load = self._check_models_to_load(models_to_load)
        self._load_models(models_to_load)

    def _check_models_to_load(self, models_to_load):
        if models_to_load is None:
            models_to_load_checked = self.all_supported_models
        else:
            models_to_load_checked = []
            for model_name in models_to_load:
                if model_name in self.all_supported_models:
                    models_to_load_checked.append(model_name)
                else:
                    warnings.warn(f"Model not supported: {model_name}")

        return models_to_load_checked

    def _load_models(self, models_to_load):
        # Load the models
        if "USE" in models_to_load:
            logger.info("Loading USE...")
            use_version = 5
            use_url = "https://tfhub.dev/google/universal-sentence-encoder-large"
            self.models["USE"] = hub.load(f"{use_url}/{use_version}")

        if "SBERT" in models_to_load:
            logger.info("Loading SBERT...")
            self.models["SBERT"] = SentenceTransformer('bert-base-nli-mean-tokens')

        if "BSV" in models_to_load:
            logger.info("Loading BioSentVec...")
            bsv_path = self.assets_path / 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
            self.models["BSV"] = sent2vec.Sent2vecModel()
            self.models["BSV"].load_model(str(bsv_path))
        if "SBIOBERT" in models_to_load:
            logger.info("Loading SBioBERT...")
            self.models["SBIOBERT"] = SBioBERT()

    def _bsv_preprocess(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens = [token for token in word_tokenize(text)
                  if token not in string.punctuation and token not in self.bsv_stopwords]
        return ' '.join(tokens)

    def embed_sentences(self, sentences, model_name):
        """Embed given sentences.

        Parameters
        ----------
        sentences : list
            A list of sentences to embed. Each sentence should be a string.
        model_name : str
            The name of the embedding model to use.

        Returns
        -------
        embeddings : numpy.ndarray
            The embeddings of the sentences with the shape
            (n_sentences, dim_embedding).
        """
        if model_name.upper() not in self.models:
            raise ValueError(f"Model f{model_name} is not available.")

        model_name = model_name.upper()
        model = self.models[model_name]
        if model_name == "USE":
            embeddings = model(sentences).numpy()
        elif model_name in ["SBERT", "SBIOBERT"]:
            embeddings = np.stack(model.encode(sentences), axis=0)
        elif model_name == "BSV":
            preprocessed = [self._bsv_preprocess(x) for x in sentences]
            embeddings = model.embed_sentences(preprocessed)
        else:
            raise ValueError(f"Model f{model_name} is not available.")

        return embeddings

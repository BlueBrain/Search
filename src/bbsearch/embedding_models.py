"""The `EmbeddingModels` class."""

from collections import defaultdict
import string
import logging
import pathlib
import sqlite3
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

        self.synonyms_index = self._load_synonyms()

    def _load_synonyms(self):
        logger.info("Processing synonyms...")
        synonyms_path = self.assets_path / 'synonyms_list.txt'
        synonyms_dict = dict()
        with open(synonyms_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip().lower()
                if len(line) > 0:
                    words = [word.strip() for word in line.split('=')]
                    synonyms_dict[words[0]] = words[1:]

        del synonyms_dict['sars']

        synonyms_index = {x.lower(): k.lower()
                          for k, v in synonyms_dict.items() for x in v}

        return synonyms_index

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

    @staticmethod
    def sent_preprocessing(sentences, synonyms_index):
        """Pre-processing of the sentences. (Lower + Split + Replace Synonym)

        Parameters
        ----------
        sentences : List[str]
            List of N strings.
        synonyms_index: dict
            Dictionary containing as key the synonym term and as values
            the reference of this term.
        """

        return [" ".join(synonyms_index.get(y, y) for y in word_tokenize(x.lower()))
                for x in sentences]

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
            raise ValueError(f"Model {model_name} is not available.")

        if isinstance(sentences, str):
            sentences = [sentences]

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

    def compute_sentences_embeddings(self,
                                     database_path,
                                     model_name,
                                     batch_size=1000,
                                     synonym_merging=False):
        """Compute Sentences Embeddings for a given database.

        Parameters
        ----------
        database_path: pathlib.Path
            Path to the database with all the sentences to embed.
        model_name: str
            Name of the model used for the embeddings computation.
        batch_size: int
            Number of sentences embeddings computed per batch.
        synonym_merging: bool
            If True, synonyms will be merged according to the synonym list. Otherwise, nothing happens.

        Returns
        -------
        all_embeddings_and_ids: dict
            Dictionary containing the concatenated numpy array (sentence_id, embeddings)
        """
        embeddings = defaultdict(list)
        all_embeddings_and_ids = dict()
        all_ids = []

        with sqlite3.connect(str(database_path)) as db:
            curs = db.cursor()
            curs.execute("""SELECT sentence_id, text FROM sentences
                        WHERE sha IN (SELECT sha FROM article_id_2_sha WHERE article_id IN
                        (SELECT article_id FROM articles WHERE has_covid19_tag is True))""")
            while True:
                batch = curs.fetchmany(batch_size)
                if not batch:
                    break
                ids, sentences = zip(*batch)

                all_ids.extend(ids)

                if synonym_merging:
                    sentences = self.sent_preprocessing(sentences, self.synonyms_index)

                tmp_embeddings_ = self.embed_sentences(sentences, model_name=model_name)
                embeddings[model_name].append(tmp_embeddings_)

        all_embeddings = np.concatenate(embeddings[model_name], axis=0)
        all_ids = np.array(all_ids).reshape((-1, 1))
        all_embeddings_and_ids[model_name] = np.concatenate((all_ids, all_embeddings), axis=1)

        return all_embeddings_and_ids

    def save_sentence_embeddings(self,
                                 database_path,
                                 saving_directory=None,
                                 synonym_merging=False):
        """Saves the sentences embeddings for a database.

        Parameters
        ----------
        database_path: pathlib.Path
            Path to the database with all the sentences to embed.
        saving_directory: str
            Path where the embeddings are saved.
        synonym_merging: bool
            If True, synonyms will be merged according to the synonym list. Otherwise, nothing happens.
        """
        saving_directory = saving_directory or pathlib.Path.cwd()

        if not database_path.exists():
            raise ValueError(f'The database {database_path} does not exist!')

        for model_name in self.models.keys():
            all_embeddings_and_ids = self.compute_sentences_embeddings(database_path=database_path,
                                                                       model_name=model_name,
                                                                       synonym_merging=synonym_merging)
            if synonym_merging:
                file_name = pathlib.Path(saving_directory) / f"{model_name}_sentence_embeddings_merged_synonyms.npz"
            else:
                file_name = pathlib.Path(saving_directory) / f"{model_name}_sentence_embeddings.npz"

            np.savez_compressed(file=str(file_name), **all_embeddings_and_ids)

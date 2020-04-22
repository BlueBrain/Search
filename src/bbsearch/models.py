from string import punctuation
import logging

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import tensorflow_hub as hub
import sent2vec
from sentence_transformers import SentenceTransformer

from .s_bio_bert import SBioBERT

logger = logging.getLogger(__name__)


class AllModels:

    def __init__(self, all_data):
        self.assets_path = all_data.assets_path
        self.embeddings_path = all_data.embeddings_path

        logger.info("Loading USE...")
        self.use_version = 5
        use_url = "https://tfhub.dev/google/universal-sentence-encoder-large"
        self.use = hub.load(f"{use_url}/{self.use_version}")

        logger.info("Loading SBERT...")
        self.sbert = SentenceTransformer('bert-base-nli-mean-tokens')

        logger.info("Loading BioSentVec...")
        self.bsv = sent2vec.Sent2vecModel()
        self.bsv.load_model(str(self.assets_path / 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'))
        self.bsv_stopwords = set(stopwords.words('english'))

        logger.info("Loading SBioBERT...")
        self.sbiobert = SBioBERT()

        logger.info("Processing synonyms...")
        self.synonyms_dict = dict()
        with open(self.assets_path / 'synonyms_list.txt', 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip().lower()
                if len(line) > 0:
                    words = [word.strip() for word in line.split('=')]
                    self.synonyms_dict[words[0]] = words[1:]

        del self.synonyms_dict['sars']

        self.synonyms_index = {x.lower(): k.lower()
                               for k, v in self.synonyms_dict.items() for x in v}

        logger.info("Loading sentence embeddings from disk...")
        self.EMBEDDINGS_NAMES = ['USE', 'SBERT', 'BSV', 'SBIOBERT']

        self.embeddings = dict()
        for embeddings_name in self.EMBEDDINGS_NAMES:
            logger.info(f"> {embeddings_name}")
            current_embedding_name = f'{embeddings_name}_sentence_embeddings.npz'
            current_embedding_path = self.embeddings_path / current_embedding_name
            current_embedding = np.load(current_embedding_path)[embeddings_name]
            self.embeddings[embeddings_name] = current_embedding

        self.embeddings_syns = dict()
        for embeddings_name in self.EMBEDDINGS_NAMES:
            logger.info(f"> {embeddings_name}, merged synonyms")
            current_embedding_name = f'{embeddings_name}_sentence_embeddings_merged_synonyms.npz'
            current_embedding_path = self.embeddings_path / current_embedding_name
            current_embedding = np.load(current_embedding_path)[embeddings_name]
            self.embeddings_syns[embeddings_name] = current_embedding

    def bsv_preprocess(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens = [token for token in word_tokenize(text)
                  if token not in punctuation and token not in self.bsv_stopwords]
        return ' '.join(tokens)

    @staticmethod
    def sent_preprocessing(sentences, synonyms_index):
        """Preprocessing of the sentences. (Lower + Split + Replace Synonym)

        Parameters
        ----------
        sentences : List[str]
            List of N strings.
        synonyms_index: dict
            Dictionary containing as key the synonym term and as values the reference of this term.
        """

        return [" ".join(synonyms_index.get(y, y) for y in word_tokenize(x.lower()))
                for x in sentences]

    def embed_sentences(self, sentences, embedding_name, embedding_model):
        if embedding_name == 'USE':
            return embedding_model(sentences).numpy()

        elif embedding_name == 'SBERT':
            return np.stack(embedding_model.encode(sentences), axis=0)

        elif embedding_name == 'SBIOBERT':
            return np.stack(embedding_model.encode(sentences), axis=0)

        elif embedding_name == 'BSV':
            preprocessed = [self.bsv_preprocess(x) for x in sentences]
            return embedding_model.embed_sentences(preprocessed)

        else:
            raise NotImplementedError(
                f'Embedding {repr(embedding_name)} not available!')

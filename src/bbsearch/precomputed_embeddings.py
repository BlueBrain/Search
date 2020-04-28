import logging
import pathlib
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class PrecomputedEmbeddings:

    def __init__(self, embeddings_path, models_to_load=None, load_with_merged_synonyms=False):
        self.embeddings_path = pathlib.Path(embeddings_path)
        self.load_with_merged_synonyms = load_with_merged_synonyms
        self.all_supported_models = ["USE", "SBERT", "BSV", "SBIOBERT"]
        self.embeddings = dict()
        self.embeddings_syns = dict()

        models_to_load = self._check_models_to_load(models_to_load)
        self._load_all_embeddings(models_to_load)

    def _load_all_embeddings(self, models_to_load):
        logger.info("Loading sentence embeddings from disk...")
        for model_name in models_to_load:
            self.embeddings[model_name] = self._load_embedding(model_name)
            if self.load_with_merged_synonyms:
                self.embeddings_syns[model_name] = self._load_embedding(
                    model_name, merged_synonyms=True)

    def _load_embedding(self, model_name, merged_synonyms=False):
        logger.info(f"> {model_name}, merged_synonyms={merged_synonyms}")
        if not merged_synonyms:
            file_name = f'{model_name}_sentence_embeddings.npz'
        else:
            file_name = f'{model_name}_sentence_embeddings_merged_synonyms.npz'
        file_path = self.embeddings_path / file_name
        embeddings = np.load(file_path)[model_name]

        return embeddings

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

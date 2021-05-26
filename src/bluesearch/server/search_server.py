"""The search server."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pathlib

import torch
from flask import Flask, jsonify, request

import bluesearch

from ..embedding_models import EmbeddingModel, get_embedding_model
from ..search import SearchEngine
from ..utils import H5


class SearchServer(Flask):
    """The BBS search server.

    Parameters
    ----------
    trained_models_path : str or pathlib.Path
        The folder containing pre-trained models.
    embeddings_h5_path : str or pathlib.Path
        The path to the h5 file containing pre-computed embeddings.
    connection : sqlalchemy.engine.Engine
        The database connection.
    indices : np.ndarray
        1D array containing sentence_ids to be considered for precomputed embeddings.
    models : list_like
        A list of model names of the embedding models to load.
    """

    def __init__(
        self,
        trained_models_path,
        embeddings_h5_path,
        indices,
        connection,
        models,
    ):
        package_name, *_ = __name__.partition(".")
        super().__init__(import_name=package_name)

        self.version = bluesearch.__version__
        self.server_name = "SearchServer"
        self.connection = connection

        if indices is None:
            raise ValueError("Please specify the indices.")

        self.indices = indices
        self.logger.info("Initializing the server...")
        self.logger.info(f"Name: {self.server_name}")
        self.logger.info(f"Version: {self.version}")

        self.trained_models_path = pathlib.Path(trained_models_path)
        self.embeddings_h5_path = pathlib.Path(embeddings_h5_path)

        self.logger.info("Initializing embedding models...")
        self.models = models
        self.embedding_models = {
            model_name: self._get_model(model_name) for model_name in models
        }

        self.logger.info("Loading precomputed embeddings...")
        # here we're assuming that all embeddings (up to the 0th row)
        # are correctly populated, note the `[1:]` slice.
        self.precomputed_embeddings = {
            model_name: H5.load(
                self.embeddings_h5_path,
                model_name,
            )[1:]
            for model_name in self.embedding_models
        }

        self.logger.info("Normalizing precomputed embeddings...")
        for model_name, embeddings in self.precomputed_embeddings.items():
            embeddings_t = torch.from_numpy(embeddings)
            norm = torch.norm(input=embeddings_t, dim=1, keepdim=True)
            norm[norm == 0] = 1
            embeddings_t /= norm
            self.precomputed_embeddings[model_name] = embeddings_t

        self.logger.info("Constructing the search engine...")
        self.search_engine = SearchEngine(
            self.embedding_models,
            self.precomputed_embeddings,
            self.indices,
            self.connection,
        )

        self.add_url_rule("/help", view_func=self.help, methods=["POST"])
        self.add_url_rule("/", view_func=self.query, methods=["POST"])

        self.logger.info("Initialization done.")

    def _get_model(self, model_name: str) -> EmbeddingModel:
        """Construct an embedding model from its name.

        Parameters
        ----------
        model_name :
            The name of the model.

        Returns
        -------
        sentence_embedding_model : EmbeddingModel
            The sentence embedding model instance.
        """
        biobert_nli_sts_cord19_v1_model_name = "biobert_nli_sts_cord19_v1"
        biobert_nli_sts_cord19_v1_model_path = (
            self.trained_models_path / biobert_nli_sts_cord19_v1_model_name
        )

        model_factories = {
            "SBioBERT": lambda: get_embedding_model("SBioBERT"),
            "SBERT": lambda: get_embedding_model("SBERT"),
            "BioBERT NLI+STS": lambda: get_embedding_model("BioBERT NLI+STS"),
            "BioBERT NLI+STS CORD-19 v1": lambda: get_embedding_model(
                "SentTransformer", checkpoint_path=biobert_nli_sts_cord19_v1_model_path
            ),
        }

        if model_name not in model_factories:
            raise ValueError(f"Unknown model name: {model_name}")
        selected_factory = model_factories[model_name]

        return selected_factory()

    def help(self):
        """Help the user by sending information about the server."""
        self.logger.info("Help called")

        response = {
            "name": self.server_name,
            "version": self.version,
            "database": self.connection.url.database,
            "supported_models": self.models,
            "description": "Run the BBS text search for a given sentence.",
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json",
                },
                "/": {
                    "description": "Compute search through database"
                    "and give back most similar sentences to the query.",
                    "response_content_type": "application/json",
                    "required_fields": {
                        "query_text": [],
                        "which_model": self.models,
                        "k": "integer number",
                    },
                    "accepted_fields": {
                        "is_english": [True, False],
                        "has_journal": [True, False],
                        "data_range": ("start_date", "end_date"),
                        "deprioritize_strength": [
                            "None",
                            "Weak",
                            "Mild",
                            "Strong",
                            "Stronger",
                        ],
                        "exclusion_text": [],
                        "inclusion_text": [],
                        "deprioritize_text": [],
                    },
                },
            },
        }

        return jsonify(response)

    def query(self):
        """Respond to a query.

        The main query callback routed to "/".

        Returns
        -------
        response_json : flask.Response
            The JSON response to the query.
        """
        self.logger.info("Search query received")
        if request.is_json:
            self.logger.info("Search query is JSON. Processing.")
            json_request = request.get_json()

            which_model = json_request.pop("which_model")
            k = json_request.pop("k")
            query_text = json_request.pop("query_text")

            self.logger.info("Search parameters:")
            self.logger.info(f"which_model: {which_model}")
            self.logger.info(f"k          : {k}")
            self.logger.info(f"query_text : {query_text}")

            self.logger.info("Starting the search...")
            sentence_ids, similarities, stats = self.search_engine.query(
                which_model, k, query_text, **json_request
            )

            self.logger.info(f"Search completed, got {len(sentence_ids)} results.")

            response = {
                "sentence_ids": sentence_ids.tolist(),
                "similarities": similarities.tolist(),
                "stats": stats,
            }
        else:
            self.logger.info("Search query is not JSON. Not processing.")
            response = {
                "sentence_ids": None,
                "similarities": None,
                "stats": None,
            }

        response_json = jsonify(response)

        return response_json

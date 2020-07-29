"""The search server."""
import logging
import pathlib

from flask import request, jsonify
import numpy as np

import bbsearch
from ..embedding_models import BSV, SBioBERT
from ..search import LocalSearcher


class SearchServer:
    """The BBS search server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    trained_models_path : str or pathlib.Path
        The folder containing pre-trained models.
    embeddings_path : str or pathlib.Path
        The folder containing pre-computed embeddings.
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        The database connection.
        Version.
    """

    def __init__(self,
                 app,
                 trained_models_path,
                 embeddings_path,
                 connection):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = bbsearch.__version__
        self.name = "SearchServer"
        self.app = app
        self.connection = connection

        self.logger.info("Initializing the server...")
        self.logger.info(f"Name: {self.name}")
        self.logger.info(f"Version: {self.version}")

        trained_models_path = pathlib.Path(trained_models_path)
        embeddings_path = pathlib.Path(embeddings_path)

        self.logger.info("Initializing embedding models...")
        bsv_model_name = "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        bsv_model_path = trained_models_path / bsv_model_name
        embedding_models = {
            "BSV": BSV(checkpoint_model_path=bsv_model_path),
            "SBioBERT": SBioBERT()
        }

        self.logger.info("Loading precomputed embeddings...")
        precomputed_embeddings = {
            model_name: np.load(embeddings_path / f"{model_name}.npy").astype(np.float32)
            for model_name in embedding_models
        }

        self.local_searcher = LocalSearcher(
            embedding_models, precomputed_embeddings, self.connection)

        app.route("/help", methods=["POST"])(self.help)
        app.route("/", methods=["POST"])(self.query)

        self.logger.info("Initialization done.")

    def help(self):
        """Help the user by sending information about the server."""
        self.logger.info("Help called")

        response = {
            "name": self.name,
            "version": self.version,
            "description": "Run the BBS text search for a given sentence.",
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json"
                },
                "/": {
                    "description": "Compute search through database"
                                   "and give back most similar sentences to the query.",
                    "response_content_type": "application/json",
                    "required_fields": {
                        "query_text": [],
                        "which_model": ["BSV", "SBioBERT"],
                        "k": 'integer number'
                    },
                    "accepted_fields": {
                        "has_journal": [True, False],
                        "data_range": ('start_date', 'end_date'),
                        "deprioritize_strength": ['None', 'Weak', 'Mild',
                                                  'Strong', 'Stronger'],
                        "exclusion_text": [],
                        "deprioritize_text": []
                    }
                }
            }
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
            sentence_ids, similarities, stats = self.local_searcher.query(
                which_model,
                k,
                query_text,
                **json_request)

            self.logger.info(f"Search completed, got {len(sentence_ids)} results.")

            response = dict(
                sentence_ids=sentence_ids.tolist(),
                similarities=similarities.tolist(),
                stats=stats)
        else:
            self.logger.info("Search query is not JSON. Not processing.")
            response = dict(
                sentence_ids=None,
                similarities=None,
                stats=None)

        response_json = jsonify(response)

        return response_json

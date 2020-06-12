"""The search server."""
import logging
import pathlib

from flask import request, jsonify
import numpy as np

from ..embedding_models import BSV, SBioBERT
from ..search import LocalSearcher

logger = logging.getLogger(__name__)


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
    database_path : str or pathlib.Path
        The path to where the database file is.
    """

    def __init__(self,
                 app,
                 trained_models_path,
                 embeddings_path,
                 database_path):

        self.version = "1.0"
        self.name = "SearchServer"
        self.app = app

        trained_models_path = pathlib.Path(trained_models_path)
        embeddings_path = pathlib.Path(embeddings_path)

        logger.info("Initializing embedding models...")
        bsv_model_name = "BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        bsv_model_path = trained_models_path / bsv_model_name
        embedding_models = {
            "BSV": BSV(checkpoint_model_path=bsv_model_path),
            "SBioBERT": SBioBERT()
        }

        logger.info("Loading precomputed embeddings...")
        precomputed_embeddings = {
            model_name: np.load(embeddings_path / f"{model_name}.npy").astype(np.float32)
            for model_name in embedding_models
        }

        self.local_searcher = LocalSearcher(
            embedding_models, precomputed_embeddings, database_path)

        app.route("/help", methods=["POST"])(self.help)
        app.route("/", methods=["POST"])(self.query)
        logger.info("Server initialization done.")

    def help(self):
        """Help the user by sending information about the server."""
        response = {
            "name": self.name,
            "version": self.version,
            "description": "Run the BBS text search for a given sentence.",
            "required fields": ["text", "model", "k"],
            "optional fields": ['has_journal', 'date_range', 'deprioritize_strength',
                                'exclusion_text', 'deprioritize_text']
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
        if request.is_json:
            json_request = request.get_json()

            which_model = json_request.pop("which_model")
            k = json_request.pop("k")
            query_text = json_request.pop("query_text")

            sentence_ids, similarities, stats = self.local_searcher.query(
                which_model,
                k,
                query_text,
                **json_request)

            response = dict(
                sentence_ids=sentence_ids.tolist(),
                similarities=similarities.tolist(),
                stats=stats)
        else:
            response = dict(
                sentence_ids=None,
                similarities=None,
                stats=None)

        response_json = jsonify(response)

        return response_json

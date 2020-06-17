"""The mining server."""
from collections import defaultdict
import io
import logging
import pathlib

from flask import jsonify, make_response, request
import spacy

from ..mining import ChemProt, TextMiningPipeline


logger = logging.getLogger(__name__)

ENTITY_TYPES = {'en_ner_craft_md': ['GGP', 'SO', 'TAXON', 'CHEBI', 'GO', 'CL']}


class MiningServer:
    """The BBS mining server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    models_path : str or pathlib.Path
        The folder containing pre-trained models.
    """

    def __init__(self, app, models_path):
        self.version = "1.0"
        self.name = "MiningServer"
        self.models_path = pathlib.Path(models_path)

        self.app = app
        self.app.route("/", methods=["POST"])(self.pipeline)
        self.app.route("/help", methods=["POST"])(self.help)

        # Entities Extractors (EE)
        self.all_ee_models = {'en_ner_craft_md': spacy.load("en_ner_craft_md")}

        # Relations Extractors (RE)
        chemprot_model_path = self.models_path / 'scibert_chemprot.tar.gz'
        self.all_re_models = {'chemprot': (('CHEBI', 'GGP'), ChemProt(chemprot_model_path))}

        # Full Pipeline
        self.text_mining_pipeline = None

    def help(self):
        """Help the user by sending information about the server."""
        response = {
            "name": self.name,
            "version": self.version,
            "description": "Run the BBS text mining pipeline on a given text.",
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json"
                },
                "/": {
                    "description": "Compute text mining pipeline"
                                   "(extract entities and relations).",
                    "response_content_type": "text/csv",
                    "required_fields": {
                        "text": [],
                        "ee_models": ['en_ner_craft_md'],
                    },
                    "accepted_fields": {
                        "article_id": [],
                        "re_models": ['', 'chemprot'],
                        "return_prob": [True, False],
                        "debug": [True, False]
                    }
                }
            }
        }

        return jsonify(response)

    @staticmethod
    def make_error_response(error_message):
        """Create response if there is an error during the process.

        Parameters
        ----------
        error_message: str
            Error message to send if there is an issue.

        Returns
        -------
        response: str
            Response to send with the error_message in a json format.
        """
        response = jsonify({"error": error_message})

        return response

    def models_selection(self, requested_ee_models, requested_re_models):
        """Select ee_models and re_models for a given request.

        Parameters
        ----------
        requested_ee_models: str
            List of models name to keep for the entity extraction.
            Models name have to be comma separated.
        requested_re_models: str
            List of models name to keep for the relation extraction.
            Models name have to be comma separated.
            If this string is empty, we consider that the user does not want to
            do relation extraction.

        Returns
        -------
        selected_ee_models: list()
            List of all the entity extraction models selected.
        selected_re_models: dict()
            Dictionary of all the entity extraction models selected.
        """
        error = None
        selected_ee_models = None
        selected_re_models = None
        entity_types = set()

        if not requested_ee_models:
            error = f'EE models needs to be specified. Here are the available models: ' \
                    f'{list(self.all_ee_models.keys())}'
            return selected_ee_models, selected_re_models, error

        else:
            ee_model_names = set(requested_ee_models.split(','))
            try:
                selected_ee_models = [self.all_ee_models[model_name.strip()] for model_name in ee_model_names]
            except KeyError:
                error = f'The available entities extraction models are {list(self.all_ee_models.keys())}'
                return selected_ee_models, selected_re_models, error

        for model_name in ee_model_names:
            entity_types |= set(ENTITY_TYPES[model_name])

        if not requested_re_models:
            selected_re_models = {}
        else:
            re_model_names = set(requested_re_models.split(','))
            selected_re_models = defaultdict(list)
            for model_name in re_model_names:
                try:
                    ee_couple, model_instance = self.all_re_models[model_name]
                    if set(ee_couple).issubset(entity_types):
                        selected_re_models[ee_couple].append(model_instance)
                except KeyError:
                    error = f'The model {model_name} does not exists.' \
                            f'The available relations extraction models are {list(self.all_re_models.keys())}'
                    return selected_ee_models, selected_re_models, error

        return selected_ee_models, selected_re_models, error

    def pipeline(self):
        """Respond to a mining query."""
        if request.is_json:
            json_request = request.get_json()
            text = json_request.get("text")
            requested_ee_models = json_request.get("ee_models", '')
            requested_re_models = json_request.get("re_models", '')
            article_id = json_request.get("article_id")
            return_prob = json_request.get("return_prob", False)
            debug = json_request.get("debug", False)

            if text is None:
                response = self.make_error_response("The request text is missing.")
                return response
            else:
                selected_ee_models, selected_re_models, error = self.models_selection(requested_ee_models,
                                                                                      requested_re_models)
                if error is None:
                    self.text_mining_pipeline = TextMiningPipeline(selected_ee_models,
                                                                   selected_re_models)
                    df = self.text_mining_pipeline(
                        text=text,
                        article_id=article_id,
                        return_prob=return_prob,
                        debug=debug)

                    with io.StringIO() as csv_file_buffer:
                        df.to_csv(csv_file_buffer, index=False)
                        response = make_response(csv_file_buffer.getvalue())
                    response.headers["Content-Type"] = "text/csv"
                    response.headers["Content-Disposition"] = "attachment; filename=bbs_mining_results.csv"
                else:
                    response = self.make_error_response(error)

        else:
            response = self.make_error_response("The request has to be a JSON object.")

        return response

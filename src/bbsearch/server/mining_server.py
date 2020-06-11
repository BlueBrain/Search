import io
import logging
import pathlib

from flask import jsonify, make_response, request
import pandas as pd
import spacy

from ..mining import ChemProt, TextMiningPipeline


logger = logging.getLogger(__name__)


class MiningServer:

    def __init__(self, app, models_path):
        self.version = "1.0"
        self.name = "MiningServer"
        self.models_path = pathlib.Path(models_path)

        self.app = app
        self.app.route("/", methods=["POST"])(self.pipeline)
        self.app.route("/help", methods=["POST"])(self.help)

        # Entities Extractors (EE)
        ee_model = spacy.load("en_ner_craft_md")

        # Relations Extractors (RE)
        chemprot_model_path = self.models_path / 'scibert_chemprot.tar.gz'
        re_models = {('CHEBI', 'GGP'): [ChemProt(chemprot_model_path)]}

        # Full Pipeline
        self.text_mining_pipeline = TextMiningPipeline(ee_model, re_models)

    def help(self):
        response = {
            "name": self.name,
            "version": self.version,
            "models_path": str(self.models_path),
            "description": "Run the BBS text mining pipeline on a given text.",
            "required fields": ["text"],
            "optional fields": ["article_id", "return_prob", "debug"],
        }

        return jsonify(response)

    @staticmethod
    def make_error_response(error_message):
        response = {"error": error_message}

        return jsonify(response)

    def pipeline(self):
        if request.is_json:
            json_request = request.get_json()
            text = json_request.get("text")
            article_id = json_request.get("article_id")
            return_prob = json_request.get("return_prob") or False
            debug = json_request.get("debug") or False

            if text is None:
                response = self.make_error_response("The request text is missing.")
            else:
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
            response = self.make_error_response("The request has to be a JSON object.")

        return response



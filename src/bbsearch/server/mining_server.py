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
            "mandatory fields": ["text"],
            "optional fields": ["article_id", "return_prob", "debug"]
        }

        return jsonify(response)

    def pipeline(self):
        if request.is_json:
            json_request = request.get_json()
            text = json_request.get("text")
            article_id = json_request.get("article_id")
            return_prob = json_request.get("return_prob") or False
            debug = json_request.get("debug") or False

            if text is None:
                response = jsonify({
                    "error": "The request text is empty"
                })
            else:
                df = self.text_mining_pipeline(
                    text=text,
                    article_id=article_id,
                    return_prob=return_prob,
                    debug=debug)

                csv_file_buffer = io.StringIO()
                df.to_csv(csv_file_buffer, index=False)

                response = make_response(csv_file_buffer.getvalue())
                response.headers["Content-Disposition"] = "attachment; filename=mining_results.csv"
                response.headers["Content-Type"] = "text/csv"

        else:
            response = jsonify({
                "error": "The request has to be a JSON object."
            })

        return response



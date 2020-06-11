import io
import logging

from flask import jsonify, make_response
import pandas as pd
import spacy

from ..mining import ChemProt, TextMiningPipeline


logger = logging.getLogger(__name__)


class MiningServer:

    def __init__(self, app, models_path):
        self.version = "1.0"
        self.name = "MiningServer"

        self.app = app
        self.app.route("/", methods=["POST"])(self.pipeline)
        self.app.route("/identify", methods=["POST"])(self.identify)

        # Entities Extractors (EE)
        ee_model = spacy.load("en_ner_craft_md")

        # Relations Extractors (RE)
        chemprot_model_path = models_path / 'scibert_chemprot.tar.gz'
        re_models = {('CHEBI', 'GGP'): [ChemProt(chemprot_model_path)]}

        # Full Pipeline
        self.text_mining_pipeline = TextMiningPipeline(ee_model, re_models)

    def identify(self):
        response = {
            "name": self.name,
            "version": self.version,
        }

        return jsonify(response)

    def pipeline(self):
        df = pd.DataFrame([
            {"entity_type": "DRUG", "entity": "paracetamol"},
            {"entity_type": "ORGAN", "entity": "heart"}
        ])

        csv_file_buffer = io.StringIO()
        df.to_csv(csv_file_buffer, index=False)

        response = make_response(csv_file_buffer.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=mining_results.csv"
        response.headers["Content-Type"] = "text/csv"

        return response

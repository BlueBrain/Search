"""The mining server."""
import io
import logging
import pathlib
import sqlite3

from flask import jsonify, make_response, request
import pandas as pd
import spacy

from ..mining import run_pipeline

logger = logging.getLogger(__name__)


class MiningServer:
    """The BBS mining server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    models_path : str or pathlib.Path
        The folder containing pre-trained models.
    database_path : str or pathlib.Path
        Path to the sql database.
    """

    def __init__(self, app, models_path, database_path):
        self.version = "1.0"
        self.name = "MiningServer"
        self.models_path = pathlib.Path(models_path)
        self.database_path = pathlib.Path(database_path)

        self.app = app
        self.app.route("/text", methods=["POST"])(self.pipeline_text)
        self.app.route("/database", methods=["POST"])(self.pipeline_database)
        self.app.route("/help", methods=["POST"])(self.help)

        # Entities Extractors (EE)
        self.ee_model = spacy.load("en_ner_craft_md")

        # Relations Extractors (RE)
        self.re_models = {}

    def help(self):
        response = {
            "name": self.name,
            "version": self.version,
            "description": "Run the BBS text mining pipeline on a given text.",
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json"
                },
                "/text": {
                    "description": "Extract entities and relations from a given text.",
                    "response_content_type": "text/csv",
                    "required_fields": {
                        "text": []
                    },
                    "accepted_fields": {
                        "debug": [True, False]
                    }
                },
                "/database": {
                    "description": "Extract entities and relations for given paragraph ids from "
                                   "the database",
                    "response_content_type": "text/csv",
                    "required_fields": {
                        "paragraph_ids": [('paragraph_id_1', 'article_id_1'), ],
                    },
                    "accepted_fields": {
                        "debug": [True, False]
                    }
                }
            }
        }
        """Help the user by sending information about the server."""

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

    def pipeline_database(self):
        """Respond to a query on specific paragraphs in the database."""
        if request.is_json:
            json_request = request.get_json()
            paragraph_ids = json_request.get("paragraph_ids")
            debug = json_request.get("debug", False)

            if paragraph_ids is None:
                response = self.make_error_response("The request text is missing.")
            else:
                paragraph_ids_joined = ','.join(f"\"{id_}\"" for id_ in set(paragraph_ids))

                with sqlite3.connect(str(self.database_path)) as db_cnxn:
                    sql_query = f"SELECT paragraph_id, section_name, text FROM paragraphs WHERE" \
                                f" paragraph_id IN ({paragraph_ids_joined })"

                    texts_df = pd.read_sql(sql_query, db_cnxn)
                    raise NotImplementedError()
                    texts = []  # TOD
                df = run_pipeline(texts, self.ee_model, self.re_models, debug=debug)

                with io.StringIO() as csv_file_buffer:
                    df.to_csv(csv_file_buffer, index=False)
                    response = make_response(csv_file_buffer.getvalue())
                response.headers["Content-Type"] = "text/csv"
                response.headers["Content-Disposition"] = "attachment; filename=bbs_mining_results.csv"

        else:
            response = self.make_error_response("The request has to be a JSON object.")

        return response

    def pipeline_text(self):
        """Respond to a custom text query."""
        if request.is_json:
            json_request = request.get_json()
            text = json_request.get("text")
            debug = json_request.get("debug", False)

            if text is None:
                response = self.make_error_response("The request text is missing.")
            else:
                df = run_pipeline([(text, {})], self.ee_model, self.re_models, debug=debug)

                with io.StringIO() as csv_file_buffer:
                    df.to_csv(csv_file_buffer, index=False)
                    response = make_response(csv_file_buffer.getvalue())
                response.headers["Content-Type"] = "text/csv"
                response.headers["Content-Disposition"] = "attachment; filename=bbs_mining_results.csv"

        else:
            response = self.make_error_response("The request has to be a JSON object.")

        return response

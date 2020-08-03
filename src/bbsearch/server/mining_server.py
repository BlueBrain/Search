"""The mining server."""
import io
import logging
import pathlib

from flask import jsonify, make_response, request
import pandas as pd
import spacy

from ..mining import run_pipeline
from ..sql import retrieve_articles, retrieve_paragraph


class MiningServer:
    """The BBS mining server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    models_path : str or pathlib.Path
        The folder containing pre-trained models.
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        The database connection.
    """

    def __init__(self, app, models_path, connection):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = "1.0"
        self.name = "MiningServer"
        self.models_path = pathlib.Path(models_path)
        self.connection = connection

        self.logger.info("Initializing the server...")
        self.logger.info(f"Name: {self.name}")
        self.logger.info(f"Version: {self.version}")

        self.app = app
        self.app.route("/text", methods=["POST"])(self.pipeline_text)
        self.app.route("/database", methods=["POST"])(self.pipeline_database)
        self.app.route("/help", methods=["POST"])(self.help)

        # Entities Extractors (EE)
        self.ee_model = spacy.load("en_ner_craft_md")

        # Relations Extractors (RE)
        self.re_models = {}

        self.logger.info("Initialization done.")

    def help(self):
        """Respond to the help."""
        self.logger.info("Help called")

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
                        "identifiers": [('article_id_1', 'paragraph_id_1'), ],
                    },
                    "accepted_fields": {
                        "debug": [True, False]
                    }
                }
            }
        }
        """Help the user by sending information about the server."""

        return jsonify(response)

    def pipeline_database(self):
        """Respond to a query on specific paragraphs in the database."""
        self.logger.info("Query for mining of articles received")

        if request.is_json:
            json_request = request.get_json()
            identifiers = json_request.get("identifiers")
            debug = json_request.get("debug", False)

            self.logger.info("Mining parameters:")
            self.logger.info(f"identifiers : {identifiers}")
            self.logger.info(f"debug       : {debug}")

            if identifiers is None:
                self.logger.info("No identifiers were provided. Stopping.")
                response = self.make_error_response("The request identifiers is missing.")
                return response
            else:
                self.logger.info("Retrieving article texts from the database...")

                all_article_ids = []
                all_paragraphs = pd.DataFrame()
                for (article_id, paragraph_pos) in identifiers:
                    if paragraph_pos == -1:
                        all_article_ids += [article_id]
                    else:
                        paragraph = retrieve_paragraph(article_id,
                                                       paragraph_pos,
                                                       engine=self.connection)
                        all_paragraphs = all_paragraphs.append(paragraph)

                if all_article_ids:
                    articles = retrieve_articles(article_ids=all_article_ids,
                                                 engine=self.connection)
                    all_paragraphs = all_paragraphs.append(articles)

                texts = [(row['text'],
                          {'paper_id':
                           f'{row["article_id"]}:{row["section_name"]}'
                           f':{row["paragraph_pos_in_article"]}'})
                         for _, row in all_paragraphs.iterrows()]

                self.logger.info("Running the mining pipeline...")
                df = run_pipeline(texts, self.ee_model, self.re_models, debug=debug)
                self.logger.info(f"Mining completed. Mined {len(df)} items.")

                response = self.create_response(df)
        else:
            self.logger.info("Request is not JSON. Not processing.")
            response = self.make_error_response("The request has to be a JSON object.")

        return response

    def pipeline_text(self):
        """Respond to a custom text query."""
        self.logger.info("Query for mining of raw text received")
        if request.is_json:
            self.logger.info("Request is JSON. Processing.")

            json_request = request.get_json()
            text = json_request.get("text")
            debug = json_request.get("debug", False)

            self.logger.info("Mining parameters:")
            self.logger.info(f"text  : {text}")
            self.logger.info(f"debug : {debug}")

            if text is None:
                self.logger.info("No text received. Stopping.")
                response = self.make_error_response("The request text is missing.")
            else:
                self.logger.info("Running the mining pipeline...")
                df = run_pipeline([(text, {})], self.ee_model, self.re_models, debug=debug)
                self.logger.info(f"Mining completed. Mined {len(df)} items.")

                response = self.create_response(df)
        else:
            self.logger.info("Request is not JSON. Not processing.")
            response = self.make_error_response("The request has to be a JSON object.")

        return response

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

    @staticmethod
    def create_response(dataframe):
        """Create the response thanks to dataframe.

        Parameters
        ----------
        dataframe: pd.DataFrame
            DataFrame containing all the data.

        Returns
        -------
        response: requests.response
            Response containing the dataframe converted in csv table.
        """
        with io.StringIO() as csv_file_buffer:
            dataframe.to_csv(csv_file_buffer, index=False)
            response = make_response(csv_file_buffer.getvalue())
        response.headers["Content-Type"] = "text/csv"
        response.headers["Content-Disposition"] = "attachment; filename=bbs_mining_results.csv"
        return response

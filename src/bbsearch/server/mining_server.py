"""The mining server."""
import io
import logging

from flask import jsonify, make_response, request
import pandas as pd
import spacy

from ..mining import run_pipeline


class MiningServer:
    """The BBS mining server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    models_libs : dict of str
        Dictionary mapping each type of extraction ('ee' for entities, 're' for relations, 'ae' for
        attributes) to the csv file with the information on which model to use for the extraction
        of each entity, relation, or attribute type, respectively.
    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        The database connection.
    """

    def __init__(self, app, models_libs, connection):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = "1.0"
        self.name = "MiningServer"
        self.models_libs = {k: pd.read_csv(v)
                            for k, v in models_libs.items()}
        self.ee_models = {model_name: spacy.load(model_name)
                          for model_name in self.models_libs['ee']}
        self.connection = connection

        self.logger.info("Initializing the server...")
        self.logger.info(f"Name: {self.name}")
        self.logger.info(f"Version: {self.version}")

        self.app = app
        self.app.route("/text", methods=["POST"])(self.pipeline_text)
        self.app.route("/database", methods=["POST"])(self.pipeline_database)
        self.app.route("/help", methods=["POST"])(self.help)

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
                    "description": "Mine a given text according to a given schema.",
                    "response_content_type": "text/csv",
                    "required_fields": {
                        "text": [],
                        "schema": []
                    },
                    "accepted_fields": {
                        "debug": [True, False]
                    }
                },
                "/database": {
                    "description": "Mine given paragraph ids from the database according to a given"
                                   "schema.",
                    "response_content_type": "text/csv",
                    "required_fields": {
                        "identifiers": [('article_id_1', 'paragraph_id_1'), ],
                        "schema": []
                    },
                    "accepted_fields": {
                        "debug": [True, False]
                    }
                }
            }
        }
        """Help the user by sending information about the server."""

        return jsonify(response)

    def ee_models_from_request_schema(self, schema_request):
        """Return info on which model to use to mine each of the required entity types in schema."""
        schema_request = schema_request[~schema_request['property'].isna()]
        return schema_request.merge(self.models_libs['ee'], on='entity_type', how='left')[
            ['entity_type', 'model', 'entity_type_name', 'ontology_source']]

    def pipeline_database(self):
        """Respond to a query on specific paragraphs in the database."""
        self.logger.info("Query for mining of articles received")

        if request.is_json:
            json_request = request.get_json()
            identifiers = json_request.get("identifiers")
            schema = json_request.get("schema")
            debug = json_request.get("debug", False)

            self.logger.info("Mining parameters:")
            self.logger.info(f"identifiers : {identifiers}")
            self.logger.info(f"schema      : {schema}")
            self.logger.info(f"debug       : {debug}")

            args_err_response = self.check_args_not_null(identifiers=identifiers, schema=schema)
            if args_err_response:
                return args_err_response

            self.logger.info("Parsing identifiers...")
            tmp_dict = {paragraph_id: article_id for article_id, paragraph_id in identifiers}
            paragraph_ids_joined = ','.join(f"\"{id_}\"" for id_ in tmp_dict.keys())

            sql_query = f"""
            SELECT paragraph_id, section_name, text
            FROM paragraphs
            WHERE paragraph_id IN ({paragraph_ids_joined})
            """

            self.logger.info("Retrieving article texts from the database...")
            texts_df = pd.read_sql(sql_query, self.connection)
            texts = \
                [(row['text'],
                  {'paper_id':
                    f'{tmp_dict[row["paragraph_id"]]}:{row["section_name"]}:{row["paragraph_id"]}'})
                    for _, row in texts_df.iterrows()]

            df_all = self.mine_texts(texts=texts, schema_request=schema, debug=debug)
            response = self.create_response(df_all)
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
            schema = json_request.get("schema")
            debug = json_request.get("debug", False)

            self.logger.info("Mining parameters:")
            self.logger.info(f"text        : {text}")
            self.logger.info(f"schema      : {schema}")
            self.logger.info(f"debug       : {debug}")

            args_err_response = self.check_args_not_null(text=text, schema=schema)
            if args_err_response:
                return args_err_response

            texts = [(text, {})]
            df_all = self.mine_texts(texts=texts, schema_request=schema, debug=debug)
            response = self.create_response(df_all)
        else:
            self.logger.info("Request is not JSON. Not processing.")
            response = self.make_error_response("The request has to be a JSON object.")

        return response

    def mine_texts(self, texts, schema_request, debug):
        """Rune mining pipeline on a list of texts, using models implied by the schema request."""
        self.logger.info("Running the mining pipeline...")
        ee_models_info = self.ee_models_from_request_schema(schema_request)
        ee_models_info = ee_models_info[~ee_models_info.model.isna()]

        df_all = pd.DataFrame()
        for model_name, info_slice in ee_models_info.groupby('model_name'):
            ee_model = self.ee_models[model_name]
            df = run_pipeline(texts=texts,
                              model_entities=ee_model,
                              models_relations={},
                              debug=debug)
            df_all.append(
                df.replace({'entity_type': dict(zip(info_slice['entity_type_name'],
                                                    info_slice['entity_type']))}))

        self.logger.info(f"Mining completed. Mined {len(df_all)} items.")
        return df_all.sort_values(by=['paper_id', 'start_char'], ignore_index=True)

    def check_args_not_null(self, **kwargs):
        """Sanity check that arguments provided are not null. Returns False if all is good."""
        for k, v in kwargs.items():
            if v is None:
                self.logger.info(f"No \"{k}\" was provided. Stopping.")
                return self.make_error_response(f"The request \"{k}\" is missing.")
        return False

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

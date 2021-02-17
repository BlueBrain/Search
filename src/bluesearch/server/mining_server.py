"""The mining server."""

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

import io
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import spacy
from flask import Flask, jsonify, request

import bluesearch

from ..mining import SPECS, run_pipeline
from ..sql import retrieve_articles, retrieve_mining_cache, retrieve_paragraph
from ..utils import load_spacy_model


class MiningServer(Flask):
    """The BBS mining server.

    Parameters
    ----------
    models_libs : dict of str
        Dictionary mapping each type of extraction ('ee' for entities,
        're' for relations, 'ae' for attributes) to the `pd.DataFrame` with the
        information on which model to use for the extraction of each entity,
        relation, or attribute type, respectively. For 'ee', the dataframe
        should have 3 columns: 'entity_type', 'model', 'entity_type_name'.

         - 'entity_type': name of entity type, as called in the request schema
         - 'model': name of a spaCy or scispaCy model (e.g. 'en_ner_craft_md')
           or path to a custom trained spaCy model
         - 'entity_type_name': name of entity type, as called in 'model.labels'

    connection : sqlalchemy.engine.Engine
        The database connection.
    """

    def __init__(self, models_libs, connection):
        package_name, *_ = __name__.partition(".")
        super().__init__(import_name=package_name)

        self.version = bluesearch.__version__
        self.server_name = "MiningServer"

        self.logger.info("Initializing the server")
        self.logger.info(f"Name: {self.server_name}")
        self.logger.info(f"Version: {self.version}")

        self.logger.info("Loading the model libraries")
        self.models_libs = models_libs

        self.logger.info("Loading the NER models")
        self.ee_models: Dict[str, spacy.language.Language] = {}
        self.logger.debug(f"EE model library:\n{str(self.models_libs['ee'])}")
        ee_models_meta = self.models_libs["ee"][
            ["model_id", "model_path", "entity_type"]
        ]
        for model_id, model_path, entity_type in ee_models_meta.itertuples(index=False):
            if model_id in self.ee_models:
                self.logger.info(
                    f"Entity type {entity_type}: model {model_id} already loaded"
                )
            else:
                self.logger.info(f"Entity type {entity_type}: loading model {model_id}")
                self.ee_models[model_id] = load_spacy_model(model_path)

        self.connection = connection

        self.add_url_rule("/text", view_func=self.pipeline_text, methods=["POST"])
        self.add_url_rule(
            "/database", view_func=self.pipeline_database, methods=["POST"]
        )
        self.add_url_rule("/help", view_func=self.help, methods=["POST"])

        self.logger.info("Initialization done.")

    def help(self):
        """Respond to the help."""
        self.logger.info("Help called")

        response = {
            "name": self.server_name,
            "version": self.version,
            "database": self.connection.url.database,
            "description": "Run the BBS text mining pipeline on a given text.",
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json",
                },
                "/text": {
                    "description": "Mine a given text according to a given schema.",
                    "response_content_type": "application/json",
                    "required_fields": {"text": [], "schema": []},
                    "accepted_fields": {"debug": [True, False]},
                },
                "/database": {
                    "description": "The BBS text mining server." "schema.",
                    "response_content_type": "application/json",
                    "required_fields": {
                        "identifiers": [
                            ("article_id_1", "paragraph_id_1"),
                        ],
                        "schema": [],
                    },
                    "accepted_fields": {
                        "debug": [True, False],
                        "use_cache": [True, False],
                    },
                },
            },
        }
        """Help the user by sending information about the server."""

        return jsonify(response)

    def ee_models_from_request_schema(self, schema_df):
        """Find entity extraction model for entity types."""
        schema_df = schema_df[schema_df["property"].isna()]
        return schema_df.merge(self.models_libs["ee"], on="entity_type", how="left")[
            [
                "entity_type",
                "model_id",
                "model_path",
                "entity_type_name",
                "ontology_source",
            ]
        ]

    def pipeline_database(self):
        """Respond to a query on specific paragraphs in the database."""
        self.logger.info("Query for mining of articles received")

        if request.is_json:
            json_request = request.get_json()
            identifiers = json_request.get("identifiers")
            schema_str = json_request.get("schema")
            debug = json_request.get("debug", False)
            use_cache = json_request.get("use_cache", True)

            self.logger.info("Mining parameters:")
            self.logger.info(f"identifiers : {identifiers}")
            self.logger.info(f"schema      : {schema_str}")
            self.logger.info(f"debug       : {debug}")
            self.logger.info(f"use_cache   : {use_cache}")
            self.logger.info("Mining starting...")

            args_err_response = self.check_args_not_null(
                identifiers=identifiers, schema=schema_str
            )
            if args_err_response:
                return args_err_response

            schema_df = self.read_df_from_str(schema_str)
            self.logger.debug("schema_df:")
            self.logger.debug(str(schema_df))

            if use_cache:
                self.logger.info("Using cache")
                # determine which models are necessary
                ee_models_info = self.ee_models_from_request_schema(schema_df)
                etypes_na = ee_models_info[ee_models_info["model_id"].isna()][
                    "entity_type"
                ]
                model_ids = ee_models_info[~ee_models_info["model_id"].isna()][
                    "model_id"
                ].to_list()
                self.logger.debug(f"model_names = {model_ids}")

                # get cached results
                df_all = retrieve_mining_cache(identifiers, model_ids, self.connection)
                self.logger.debug(f"cached results, df_all =\n{str(df_all)}")

                # drop unwanted entity types
                requested_etypes = schema_df["entity_type"].unique()
                df_all = df_all[df_all["entity_type"].isin(requested_etypes)]
                self.logger.debug(
                    f"dropped unwanted entity types, df_all =\n{str(df_all)}"
                )

                # append the ontology source column
                os_mapping = {
                    et: os
                    for _, (et, os) in ee_models_info[
                        ["entity_type", "ontology_source"]
                    ].iterrows()
                }
                df_all["ontology_source"] = df_all["entity_type"].apply(
                    lambda x: os_mapping[x]
                )
                self.logger.debug(
                    f"appended ontology source column, df_all =\n{str(df_all)}"
                )

                # apply specs if not debug
                if not debug:
                    df_all = pd.DataFrame(df_all, columns=SPECS)
                    self.logger.debug(f"applied column specs, df_all =\n{str(df_all)}")
            else:
                self.logger.info("Not using the cache")
                all_article_ids = []
                all_paragraphs = pd.DataFrame()
                for (article_id, paragraph_pos) in identifiers:
                    if paragraph_pos == -1:
                        all_article_ids += [article_id]
                    else:
                        paragraph = retrieve_paragraph(
                            article_id, paragraph_pos, engine=self.connection
                        )
                        all_paragraphs = all_paragraphs.append(paragraph)

                if all_article_ids:
                    articles = retrieve_articles(
                        article_ids=all_article_ids, engine=self.connection
                    )
                    all_paragraphs = all_paragraphs.append(articles)

                texts = [
                    (
                        row["text"],
                        {
                            "paper_id": f'{row["article_id"]}:{row["section_name"]}'
                            f':{row["paragraph_pos_in_article"]}'
                        },
                    )
                    for _, row in all_paragraphs.iterrows()
                ]

                df_all, etypes_na = self.mine_texts(
                    texts=texts, schema_request=schema_df, debug=debug
                )
            response = self.create_response(df_all, etypes_na)
            self.logger.info(f"Mining completed, extracted {len(df_all)} elements.")
        else:
            self.logger.info("Request is not JSON. Not processing.")
            response = self.create_error_response(
                "The request has to be a JSON object."
            )

        return response

    def pipeline_text(self):
        """Respond to a custom text query."""
        self.logger.info("Query for mining of raw text received")
        if request.is_json:
            self.logger.info("Request is JSON. Processing.")

            json_request = request.get_json()
            text = json_request.get("text")
            schema_str = json_request.get("schema")
            debug = json_request.get("debug", False)

            self.logger.info("Mining parameters:")
            self.logger.info(f"text        : {text}")
            self.logger.info(f"schema      : {schema_str}")
            self.logger.info(f"debug       : {debug}")

            args_err_response = self.check_args_not_null(text=text, schema=schema_str)
            if args_err_response:
                return args_err_response

            schema_df = self.read_df_from_str(schema_str)

            texts: Iterable[Tuple[str, Dict[Any, Any]]] = [(text, {})]
            df_all, etypes_na = self.mine_texts(
                texts=texts, schema_request=schema_df, debug=debug
            )
            response = self.create_response(df_all, etypes_na)
        else:
            self.logger.info("Request is not JSON. Not processing.")
            response = self.create_error_response(
                "The request has to be a JSON object."
            )

        return response

    def mine_texts(self, texts, schema_request, debug):
        """Run mining pipeline on a given list of texts."""
        self.logger.info("Running the mining pipeline...")

        ee_models_info = self.ee_models_from_request_schema(schema_request)
        etypes_na = ee_models_info[ee_models_info["model_id"].isna()]["entity_type"]
        ee_models_info = ee_models_info[~ee_models_info["model_id"].isna()]

        df_all = pd.DataFrame()
        for model_id, info_slice in ee_models_info.groupby("model_id"):
            ee_model = self.ee_models[model_id]
            df = run_pipeline(
                texts=texts, model_entities=ee_model, models_relations={}, debug=debug
            )
            # Select only entity types for which this model is responsible
            df = df[df["entity_type"].isin(info_slice["entity_type_name"])]
            df.reset_index()

            # Set ontology source as specified in the request
            for _, row in info_slice.iterrows():
                ont_src = row["ontology_source"]
                etype_name = row["entity_type_name"]
                df.loc[df["entity_type"] == etype_name, "ontology_source"] = ont_src

            # Rename entity types using the model library info, so that
            # we match the schema request
            df = df.replace(
                {
                    "entity_type": dict(
                        zip(info_slice["entity_type_name"], info_slice["entity_type"])
                    )
                }
            )

            df_all = df_all.append(df)

        self.logger.info(f"Mining completed. Mined {len(df_all)} items.")
        return (
            df_all.sort_values(by=["paper_id", "start_char"], ignore_index=True),
            etypes_na,
        )

    def check_args_not_null(self, **kwargs):
        """Sanity check that arguments provided are not null.

        Returns False if all is good.
        """
        for k, v in kwargs.items():
            if v is None:
                self.logger.info(f'No "{k}" was provided. Stopping.')
                return self.create_error_response(f'The request "{k}" is missing.')
        return False

    @staticmethod
    def read_df_from_str(df_str, drop_duplicates=True):
        """Read a csv file from a string into a pd.DataFrame."""
        with io.StringIO(df_str) as sio:
            schema_df = pd.read_csv(sio)
        if drop_duplicates:
            schema_df = schema_df.drop_duplicates(keep="first", ignore_index=True)
        return schema_df

    @staticmethod
    def create_error_response(error_message):
        """Create response if there is an error during the process.

        Parameters
        ----------
        error_message : str
            Error message to send if there is an issue.

        Returns
        -------
        response : str
            Response to send with the error_message in a json format.
        """
        response = jsonify(error=error_message)

        return response, 400

    @staticmethod
    def create_response(df_extractions, etypes_na):
        """Create the response thanks to dataframe.

        Parameters
        ----------
        df_extractions : pd.DataFrame
            DataFrame containing all the elements extracted by text mining.
        etypes_na : list[str]
            Entity types found in the request CSV file for which no available
            model was found in the library.

        Returns
        -------
        response : requests.response
            Response containing the dataframe converted in csv table.
        """
        csv_extractions = df_extractions.to_csv(index=False)
        warnings = [
            f'No text mining model was found in the library for "{etype}".'
            for etype in etypes_na
        ]

        return jsonify(csv_extractions=csv_extractions, warnings=warnings), 200

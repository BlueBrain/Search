"""Implementation of a server that computes sentence embeddings."""

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

import csv
import io
import textwrap

from flask import Flask, jsonify, make_response, request

import bluesearch

from .invalid_usage_exception import InvalidUsage


class EmbeddingServer(Flask):
    """Wrapper class representing the embedding server.

    Parameters
    ----------
    embedding_models : dict
        Dictionary whom keys are name of embedding_models
        and values are instance of the embedding models.
    """

    def __init__(self, embedding_models):
        package_name, *_ = __name__.partition(".")
        super().__init__(import_name=package_name)

        self.server_name = "EmbeddingServer"
        self.version = bluesearch.__version__

        self.logger.info("Initializing the server...")
        self.logger.info(f"Name: {self.server_name}")
        self.logger.info(f"Version: {self.version}")

        self.add_url_rule(rule="/", view_func=self.request_welcome)
        self.add_url_rule(rule="/help", view_func=self.help, methods=["POST"])
        self.add_url_rule(
            rule="/v1/embed/<output_type>",
            view_func=self.request_embedding,
            methods=["POST"],
        )
        self.register_error_handler(InvalidUsage, self.handle_invalid_usage)

        self.embedding_models = embedding_models

        html_header = """
        <!DOCTYPE html>
        <head>
        <title>Blue Brain Search Embedding</title>
        </head>
        """
        self.html_header = textwrap.dedent(html_header).strip() + "\n\n"

        self.output_fn = {
            "csv": self.make_csv_response,
            "json": self.make_json_response,
        }

        self.logger.info("Initialization done.")

    @staticmethod
    def handle_invalid_usage(error):
        """Handle invalid usage."""
        print("Handling invalid usage!")
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def help(self):
        """Help the user by sending information about the server."""
        self.logger.info("Got query to help on /help")

        response = {
            "name": self.server_name,
            "version": self.version,
            "description": "The BBS sentence embedding server.",
            "GET": {
                "/": {
                    "description": "Get the welcome page.",
                    "response_content_type": "text/html",
                }
            },
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json",
                },
                "/v1/embed/json": {
                    "description": "Compute text embeddings.",
                    "response_content_type": "application/json",
                    "required_fields": {
                        "model": ["SBioBERT", "SBERT", "BioBERT NLI+STS"],
                        "text": [],
                    },
                },
            },
        }

        return jsonify(response)

    def request_welcome(self):
        """Generate a welcome page."""
        self.logger.info("Got query for welcome page on /")
        html = """
        <h1>Welcome to the Blue Brain Search Embedding REST API Server</h1>
        To receive a sentence embedding proceed as follows:
        <ul>
            <li>Wrap your query into a JSON file</li>
            <li>The JSON file should be of the following form:
            <pre>
            {
                "model": "&lt;embedding model name&gt;",
                "text": "&lt;text&gt;"
            }
            </pre>
            </li>
            <li>Send the JSON file to "<tt>/v1/embed/json</tt>"</li>
            <li>Receive a response as a JSON file</li>
        </ul>
        """

        return self.html_header + textwrap.dedent(html).strip() + "\n"

    def embed_text(self, model, text):
        """Embed text.

        Parameters
        ----------
        model : str
            String representing the model name.
        text : str
            Text to be embedded.

        Returns
        -------
        np.ndarray
            1D array representing the text embedding.

        Raises
        ------
        InvalidUsage
            If the model name is invalid.
        """
        try:
            model_instance = self.embedding_models[model]
            preprocessed_sentence = model_instance.preprocess(text)
            embedding = model_instance.embed(preprocessed_sentence)
            return embedding
        except KeyError:
            raise InvalidUsage(f"Model {model} is not available.")
        except RuntimeError:
            msg = f"""
            An unhandled error occurred. You may want to contact the
            developers and provide them the model name and the text
            of the query that caused this error.

            "model": "{model}"
            "text": "{text}"
            """
            raise InvalidUsage(textwrap.dedent(msg).strip())

    @staticmethod
    def make_csv_response(embedding):
        """Generate a csv response."""
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(str(n) for n in embedding)

        response = make_response(csv_file.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        response.headers["Content-type"] = "text/csv"

        return response

    @staticmethod
    def make_json_response(embedding):
        """Generate a json response."""
        json_response = {"embedding": [float(n) for n in embedding]}
        response = jsonify(json_response)

        return response

    def request_embedding(self, output_type):
        """Request embedding."""
        self.logger.info(f"Got query for embedding on /v1/embed/{output_type}")

        if output_type.lower() not in self.output_fn:
            raise InvalidUsage(f"Output type not recognized: {output_type}")
        else:
            output_fn = self.output_fn[output_type.lower()]

        if request.is_json:
            json_request = request.get_json()
            self._check_request_validity(json_request)
            model = json_request["model"]
            text = json_request["text"]
            self.logger.info("Embedding query parameters:")
            self.logger.info(f"model: {model}")
            self.logger.info(f"text: {text}")
            self.logger.info("Calling embed_text...")
            text_embedding = self.embed_text(model, text)
            self.logger.info("Embedding computed successfully.")
            return output_fn(text_embedding)
        else:
            raise InvalidUsage("Expected a JSON file")

    @staticmethod
    def _check_request_validity(json_request):
        required_keys = {"model", "text"}
        for key in required_keys:
            if key not in json_request:
                raise InvalidUsage(f"Request must contain the key '{key}'")

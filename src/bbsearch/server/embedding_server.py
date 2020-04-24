import csv
import io
import logging
import textwrap

from flask import request, jsonify, make_response
import numpy as np

from .invalid_usage_exception import InvalidUsage
from ..embedding_models import EmbeddingModels


logger = logging.getLogger(__name__)


class EmbeddingServer:

    def __init__(self, app, assets_path):
        self.app = app
        self.app.route("/")(self.request_welcome)
        self.app.route("/v1/embed/<output_type>")(self.request_embedding)
        self.app.errorhandler(InvalidUsage)(self.handle_invalid_usage)

        self.embedding_models = EmbeddingModels(assets_path)

        html_header = """
        <!DOCTYPE html>
        <head>
        <title>BBSearch Embedding</title>
        </head>
        """
        self.html_header = textwrap.dedent(html_header).strip() + "\n\n"

        self.output_fn = {
            'csv': self.make_csv_response,
            'json': self.make_json_response,
        }

    @staticmethod
    def handle_invalid_usage(error):
        print("Handling invalid usage!")
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def request_welcome(self):
        logger.info("Welcome page requested")
        html = """
        <h1>Welcome to the BBSearch Embedding REST API Server</h1>
        
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

        return self.html_header + textwrap.dedent(html).strip() + '\n'

    def embed_text(self, model, text):
        try:
            result = self.embedding_models.embed_sentences([text], model)
            embedding = result[0]
            return embedding
        except NotImplementedError as e:
            raise InvalidUsage(f"Model {model} is not available.")

    @staticmethod
    def make_csv_response(embedding):
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(str(n) for n in embedding)

        response = make_response(csv_file.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        response.headers["Content-type"] = "text/csv"

        return response

    @staticmethod
    def make_json_response(embedding):
        json_response = dict(embedding=[float(n) for n in embedding])
        response = jsonify(json_response)

        return response

    def request_embedding(self, output_type):
        logger.info("Requested Embedding")

        if output_type.lower() not in self.output_fn:
            raise InvalidUsage(f"Output type not recognized: {output_type}")
        else:
            output_fn = self.output_fn[output_type.lower()]

        if request.is_json:
            json_request = request.get_json()
            model = json_request["model"]
            text = json_request["text"]
            text_embedding = self.embed_text(model, text)
            return output_fn(text_embedding)
        else:
            raise InvalidUsage("Expected a JSON file")

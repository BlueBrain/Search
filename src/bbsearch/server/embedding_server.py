import csv
import io
import logging
import textwrap

from flask import request, jsonify, make_response
import numpy as np

from .invalid_usage_exception import InvalidUsage

logger = logging.getLogger(__name__)


class EmbeddingServer:

    def __init__(self, app):
        self.app = app
        self.app.route("/")(self.welcome_page)
        self.app.route("/v1/embed/<output_type>")(self.handle_embedding_request)
        self.app.errorhandler(InvalidUsage)(self.handle_invalid_usage)

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

    def welcome_page(self):
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
                "texts": ["&lt;text1&gt;", "&lt;text2&gt;", ...]
            }
            </pre>
            </li>
            <li>Send the JSON file to "<tt>/v1/embed</tt>"</li>
            <li>Receive a response as a CSV file</li>
        </ul>
        """

        return self.html_header + textwrap.dedent(html).strip() + '\n'

    def embed_texts(self, model, texts):
        dim_embedding = 5
        embeddings = np.random.rand(len(texts), dim_embedding)
        return embeddings

    @staticmethod
    def make_csv_response(embeddings):
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        for embedding in embeddings:
            csv_writer.writerow(str(n) for n in embedding)

        response = make_response(csv_file.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        response.headers["Content-type"] = "text/csv"

        return response

    @staticmethod
    def make_json_response(embeddings):
        json_response = {
            "embeddings": [[float(n) for n in embedding]
                           for embedding in embeddings]
        }
        response = jsonify(json_response)

        return response

    def handle_embedding_request(self, output_type):
        logger.info("Requested Embedding")

        if output_type.lower() not in self.output_fn:
            raise InvalidUsage(f"Output type not recognized: {output_type}")
        else:
            output_fn = self.output_fn[output_type.lower()]

        if request.is_json:
            json_request = request.get_json()
            model = json_request["model"]
            texts = json_request["texts"]
            text_embeddings = self.embed_texts(model, texts)
            return output_fn(text_embeddings)
        else:
            raise InvalidUsage("Expected a JSON file")

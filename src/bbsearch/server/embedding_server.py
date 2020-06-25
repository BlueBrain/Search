"""Implementation of a server that computes sentence embeddings."""
import csv
import io
import logging
import textwrap

from flask import request, jsonify, make_response

from .invalid_usage_exception import InvalidUsage

logger = logging.getLogger(__name__)


class EmbeddingServer:
    """Wrapper class representing the embedding server.

    Parameters
    ----------
    app: flask.Flask()
        Flask application
    embedding_models: dict
        Dictionary whom keys are name of embedding_models
        and values are instance of the embedding models.
    """

    def __init__(self, app, embedding_models):
        self.name = 'EmbeddingServer'
        self.version = "1.0"

        self.app = app
        self.app.route("/")(self.request_welcome)
        self.app.route("/help", methods=["POST"])(self.help)
        self.app.route("/v1/embed/<output_type>", methods=["POST"])(self.request_embedding)
        self.app.errorhandler(InvalidUsage)(self.handle_invalid_usage)

        self.embedding_models = embedding_models

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
        """Handle invalid usage."""
        print("Handling invalid usage!")
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def help(self):
        """Help the user by sending information about the server."""
        response = {
            "name": self.name,
            "version": self.version,
            "description": "Run the BBS embedding computation server for a given sentence.",
            "GET": {
                "/": {
                    "description": "Get the welcome page.",
                    "response_content_type": "text/html"
                }
            },
            "POST": {
                "/help": {
                    "description": "Get this help.",
                    "response_content_type": "application/json"
                },
                "/v1/embed/json": {
                    "description": "Compute text embeddings.",
                    "response_content_type": "application/json",
                    "required_fields": {
                                    "model": ["BSV", "SBioBERT", "SBERT", "USE"],
                                    "text": []
                    }
                }
            }
        }

        return jsonify(response)

    def request_welcome(self):
        """Generate a welcome page."""
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
        json_response = dict(embedding=[float(n) for n in embedding])
        response = jsonify(json_response)

        return response

    def request_embedding(self, output_type):
        """Request embedding."""
        logger.info("Requested Embedding")

        if output_type.lower() not in self.output_fn:
            raise InvalidUsage(f"Output type not recognized: {output_type}")
        else:
            output_fn = self.output_fn[output_type.lower()]

        if request.is_json:
            json_request = request.get_json()
            self._check_request_validity(json_request)
            model = json_request["model"]
            text = json_request["text"]
            text_embedding = self.embed_text(model, text)
            return output_fn(text_embedding)
        else:
            raise InvalidUsage("Expected a JSON file")

    @staticmethod
    def _check_request_validity(json_request):
        required_keys = {"model", "text"}
        for key in required_keys:
            if key not in json_request:
                raise InvalidUsage(f"Request must contain the key '{key}'")

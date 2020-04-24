import csv
import io
import logging
import textwrap
import sys

import argparse
from flask import Flask, request, jsonify, make_response
import numpy as np

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=12346,
                    type=int,
                    help="The server port")
print("ARGV:", sys.argv)
args = parser.parse_args(sys.argv[1:])


class EmbeddingServer:

    def __init__(self, app):
        self.app = app
        self.app.route("/")(self.welcome_page)
        self.app.route("/v1/embed/<output_type>")(self.handle_embedding_request)
        self.app.route("/test_json")(self.handle_json_request)

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
                "text": "&lt;the text to embed&gt;"
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
        print(json_response)
        response = jsonify(json_response)

        return response

    def handle_embedding_request(self, output_type):
        logger.info("Requested Embedding")

        if output_type.lower() not in self.output_fn:
            return f"Output type not recognized: {output_type}"
        else:
            output_fn = self.output_fn[output_type.lower()]

        if request.is_json:
            json_request = request.get_json()
            model = json_request["model"]
            texts = json_request["texts"]
            text_embeddings = self.embed_texts(model, texts)
            return output_fn(text_embeddings)
        else:
            return "Error: expecting a JSON request."

    @staticmethod
    def handle_json_request():
        logger.info("Requested JSON test")
        if request.is_json:
            json_request = request.get_json()
            json_response = {
                "message": "Hello from BBSearch server!",
                "original": json_request,
            }
            response = jsonify(json_response)
        else:
            response = f"""
            Try sending a JSON request!<br>
            """
            response = textwrap.dedent(response).strip()
        return response


def main():
    app = Flask("BBSearch Embedding Server")
    EmbeddingServer(app)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=True,
    )


if __name__ == "__main__":
    main()

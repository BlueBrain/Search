import argparse
import logging
import sys

from flask import Flask

from .embedding_server import EmbeddingServer


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
args = parser.parse_args(sys.argv[1:])


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

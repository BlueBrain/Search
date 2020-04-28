import argparse
import logging
import os
import pathlib

from flask import Flask

from .embedding_server import EmbeddingServer


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
args = parser.parse_args()


def main():
    # Check the assets_path
    if "ASSETS_PATH" in os.environ:
        assets_path = pathlib.Path(os.environ["ASSETS_PATH"])
        if not assets_path.exists():
            raise ValueError("ASSETS_PATH does not represent a valid path.")
        # Create Server app
        app = Flask("BBSearch Embedding Server")
        EmbeddingServer(app, assets_path)
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            debug=True,
        )
    else:
        raise ValueError("Environmental variable ASSETS_PATH not found.")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()

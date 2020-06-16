"""The entrypoint script for the mining server."""
import argparse


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--host",
                    default="0.0.0.0",
                    type=str,
                    help="The server host IP")
parser.add_argument("--port",
                    default=8080,
                    type=int,
                    help="The server port")
parser.add_argument("--models_path",
                    default="/raid/bbs_data/trained_models",
                    type=str,
                    help="The folder with pretrained models")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    from flask import Flask
    from ..server.mining_server import MiningServer

    app = Flask("BBSearch Server")
    MiningServer(app, args.models_path)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()

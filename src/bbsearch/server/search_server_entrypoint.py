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
parser.add_argument("--embeddings_path",
                    default="/raid/bbs_data/cord19_v7/embeddings",
                    type=str,
                    help="The folder with the precomputed embeddings")
parser.add_argument("--databases_path",
                    default="/raid/bbs_data/cord19_v7/databases",
                    type=str,
                    help="The folder with databases.")
args = parser.parse_args()


def main():
    from flask import Flask
    from .search_server import SearchServer

    app = Flask("BBSearch Server")
    SearchServer(app, args.models_path, args.embeddings_path, args.databases_path)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=True,
    )


if __name__ == "__main__":
    main()

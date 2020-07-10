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
parser.add_argument("--database_path",
                    default="/raid/bbs_data/cord19_v7/databases/cord19.db",
                    type=str,
                    help="The path to the database. ")
args = parser.parse_args()


def main():
    """Execute the entry point."""
    import pathlib
    from flask import Flask
    import sqlalchemy
    from ..server.mining_server import MiningServer

    app = Flask("BBS Mining Server")
    models_path = pathlib.Path(args.models_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{args.database_path}")

    MiningServer(app, models_path, engine)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()

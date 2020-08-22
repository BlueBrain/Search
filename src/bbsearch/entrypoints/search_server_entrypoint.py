"""The entrypoint script for the search server."""
import argparse
import logging
import os
import pathlib

logger = logging.getLogger(__name__)


def get_search_app():
    """Construct the search Flask app."""
    import sqlalchemy
    from flask import Flask

    from ._helper import configure_logging
    from ..server.search_server import SearchServer
    from ..utils import H5

    # Read configuration
    log_file = os.getenv("LOG_FILE")
    debug_mode = os.getenv("DEBUG", 0)
    models_path = os.getenv("MODELS_PATH")
    embeddings_path = os.getenv("EMBEDDINGS_PATH")
    models = os.getenv("MODELS")
    mysql_driver = os.getenv("MYSQL_DRIVER")
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_port = os.getenv("MYSQL_PORT")
    mysql_database = os.getenv("MYSQL_DATABASE")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")

    # Configure logging
    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    configure_logging(log_file, log_level)

    # Check configuration
    logger.info("Checking server configuration")
    if models_path is None:
        raise ValueError("The variable $MODELS_PATH must be set")
    if embeddings_path is None:
        raise ValueError("The variable $EMBEDDINGS_PATH must be set")
    if models is None:
        raise ValueError("The variable $SEARCH_MODELS must be set")
    if mysql_host is None:
        raise ValueError("The variable $MYSQL_HOST must be set")
    if mysql_port is None:
        raise ValueError("The variable $MYSQL_PORT must be set")
    if mysql_database is None:
        raise ValueError("The variable $MYSQL_DATABASE must be set")
    if mysql_user is None:
        raise ValueError("The variable $MYSQL_USER must be set")
    if mysql_password is None:
        raise ValueError("The variable $MYSQL_PASSWORD must be set")

    # Start server
    logger.info("Creating the Flask app")
    app = Flask("BBSearch Server")
    models_path = pathlib.Path(models_path)
    embeddings_path = pathlib.Path(embeddings_path)
    indices = H5.find_populated_rows(embeddings_path, "BSV")
    mysql_dialect = "mysql"
    if mysql_driver is not None:
        mysql_dialect += f"+{mysql_driver}"
    mysql_url = (
        f"${mysql_dialect}://${mysql_user}:${mysql_password}@"
        f"{mysql_host}:{mysql_port}/{mysql_database}"
    )
    engine = sqlalchemy.create_engine(mysql_url)
    models_list = [model.strip() for model in models.split(",")]

    SearchServer(app, models_path, embeddings_path, indices, engine, models_list)

    return app


def run_search_server(argv=None):
    """Run the search server from the command line.

    This starts Flask's development web server. For development
    purposes only. For production use the corresponding docker file.

    Parameters
    ----------
    argv : list_like of str
        The command line arguments.
    """
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]", description="Start the BBSear Server.",
    )
    parser.add_argument("--host", default="localhost", type=str, help="The server host")
    parser.add_argument("--port", default=8080, type=int, help="The server port")
    parser.add_argument(
        "--env-file",
        default=".env.search",
        type=str,
        help="The name of the .env file with the server configuration",
    )
    args = parser.parse_args(argv)

    # Load configuration from a .env file
    load_dotenv(dotenv_path=args.env_file)

    print("Log file:", os.getenv("LOG_FILE"))

    app = get_search_app()
    app.run(host=args.host, port=args.port, threaded=True, debug=True)


if __name__ == "__main__":
    exit(run_search_server())

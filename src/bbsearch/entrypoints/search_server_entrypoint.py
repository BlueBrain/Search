"""The entrypoint script for the search server."""
import argparse
import logging
import pathlib

logger = logging.getLogger(__name__)


def get_search_app():
    """Construct the search Flask app."""
    import sqlalchemy

    from ..server.search_server import SearchServer
    from ..utils import H5
    from ._helper import configure_logging, get_var

    # Read configuration
    log_file = get_var("BBS_SEARCH_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_SEARCH_LOG_LEVEL", str(logging.WARNING))

    models_path = get_var("BBS_SEARCH_MODELS_PATH")
    embeddings_path = get_var("BBS_SEARCH_EMBEDDINGS_PATH")
    which_models = get_var("BBS_SEARCH_MODELS")

    mysql_url = get_var("BBS_SEARCH_MYSQL_URL")
    mysql_user = get_var("BBS_SEARCH_MYSQL_USER")
    mysql_password = get_var("BBS_SEARCH_MYSQL_PASSWORD")

    # Configure logging
    configure_logging(log_file, int(log_level))

    # Initialize flask app
    logger.info("Creating the Flask app")
    models_path = pathlib.Path(models_path)
    embeddings_path = pathlib.Path(embeddings_path)
    indices = H5.find_populated_rows(embeddings_path, "BSV")
    engine_url = f"mysql://{mysql_user}:{mysql_password}@{mysql_url}"
    engine = sqlalchemy.create_engine(engine_url)
    models_list = [model.strip() for model in which_models.split(",")]

    server_app = SearchServer(
        models_path, embeddings_path, indices, engine, models_list
    )

    return server_app


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

    # Parse arguments
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]", description="Start the BBSear Server.",
    )
    parser.add_argument("--host", default="localhost", type=str, help="The server host")
    parser.add_argument("--port", default=8080, type=int, help="The server port")
    parser.add_argument(
        "--env-file",
        default="",
        type=str,
        help="The name of the .env file with the server configuration",
    )
    args = parser.parse_args(argv)

    # Load configuration from a .env file, if one is found
    load_dotenv(dotenv_path=args.env_file)

    # Construct and launch the app
    app = get_search_app()
    app.run(host=args.host, port=args.port, threaded=True, debug=True)


if __name__ == "__main__":
    exit(run_search_server())

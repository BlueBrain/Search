"""The entrypoint script for the search server."""
import logging
import pathlib

from ._helper import configure_logging, get_var, run_server


def get_search_app():
    """Construct the search flask app."""
    import sqlalchemy

    from ..server.search_server import SearchServer
    from ..utils import H5

    # Read configuration
    log_file = get_var("BBS_SEARCH_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_SEARCH_LOG_LEVEL", logging.INFO, var_type=int)

    models_path = get_var("BBS_SEARCH_MODELS_PATH")
    embeddings_path = get_var("BBS_SEARCH_EMBEDDINGS_PATH")
    which_models = get_var("BBS_SEARCH_MODELS")

    mysql_url = get_var("BBS_SEARCH_MYSQL_URL")
    mysql_user = get_var("BBS_SEARCH_MYSQL_USER")
    mysql_password = get_var("BBS_SEARCH_MYSQL_PASSWORD")

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

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
    logger.info("Search server app created successfully.")

    return server_app


def run_search_server():
    """Run the search server."""
    run_server(get_search_app, "search")


if __name__ == "__main__":
    exit(run_search_server())

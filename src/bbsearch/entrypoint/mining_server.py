"""The entrypoint script for the mining server."""
import logging
import pathlib
import sys
import tempfile

import sqlalchemy

from ..utils import DVC
from ._helper import configure_logging, get_var, run_server


def get_mining_app():
    """Construct the mining flask app."""
    from ..server.mining_server import MiningServer

    # Read configuration
    log_file = get_var("BBS_MINING_LOG_FILE", check_not_set=False)
    log_level = get_var("BBS_MINING_LOG_LEVEL", logging.INFO, var_type=int)
    db_type = get_var("BBS_MINING_DB_TYPE")

    # Configure logging
    configure_logging(log_file, log_level)
    logger = logging.getLogger(__name__)

    # Create the database engine
    logger.info("Creating the database engine")
    if db_type == "sqlite":
        sqlite_db_path = get_var("BBS_MINING_SQLITE_DB_PATH")
        sqlite_db_path = pathlib.Path(sqlite_db_path)
        if not sqlite_db_path.exists():
            sqlite_db_path.parent.mkdir(exist_ok=True, parents=True)
            sqlite_db_path.touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{sqlite_db_path}")
    elif db_type == "mysql":
        mysql_url = get_var("BBS_MINING_MYSQL_URL")
        mysql_user = get_var("BBS_MINING_MYSQL_USER")
        mysql_password = get_var("BBS_MINING_MYSQL_PASSWORD")
        engine_url = (
            f"mysql+mysqldb://{mysql_user}:{mysql_password}@{mysql_url}?charset=utf8mb4"
        )
        engine = sqlalchemy.create_engine(engine_url)
    else:
        raise ValueError(f"This is not a valid database type: {db_type}.")

    # Create the server app
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = pathlib.Path(tmpdir_name)
        tmp_csv = tmpdir / "temp.csv"

        DVC.load_ee_models_library().to_csv(tmp_csv)

        logger.info("Creating the server app")
        mining_app = MiningServer(models_libs={"ee": tmp_csv}, connection=engine)

    return mining_app


def run_mining_server():
    """Run the mining server."""
    run_server(get_mining_app, "mining")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_mining_server())

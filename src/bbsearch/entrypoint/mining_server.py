"""The entrypoint script for the mining server."""

# BBSearch is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import logging
import pathlib
import sys
import tempfile

import sqlalchemy

from ..utils import load_ee_models_library
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
        sqlite_db_path = get_var("BBS_MINING_DB_URL")
        sqlite_db_path = pathlib.Path(sqlite_db_path)
        if not sqlite_db_path.exists():
            sqlite_db_path.parent.mkdir(exist_ok=True, parents=True)
            sqlite_db_path.touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{sqlite_db_path}")
    elif db_type == "mysql":
        mysql_url = get_var("BBS_MINING_DB_URL")
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

        df_csv = load_ee_models_library()
        df_csv.to_csv(tmp_csv)

        logger.info("Creating the server app")
        mining_app = MiningServer(models_libs={"ee": tmp_csv}, connection=engine)

    return mining_app


def run_mining_server():
    """Run the mining server."""
    run_server(get_mining_app, "mining")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_mining_server())

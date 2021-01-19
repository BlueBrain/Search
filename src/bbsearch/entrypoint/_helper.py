"""Helper functions for server entry points."""

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

import argparse
import logging
import os
import sys


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Exception handler for logging.

    For more information about the parameters see
    https://docs.python.org/3/library/sys.html#sys.exc_info

    Parameters
    ----------
    exc_type
        Type of the exception.
    exc_value
        Exception instance.
    exc_traceback
        Traceback option.

    Note
    ----
    Credit: https://stackoverflow.com/a/16993115/2804645
    """
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def configure_logging(log_file=None, level=logging.WARNING):
    """Configure the logging for the server.

    Parameters
    ----------
    log_file : str or pathlib.Path, optional
        The log file. If not provided then the log will be printed in
        the terminal.
    level : int, optional
        The logging level. See the `logging` module for more information.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="{asctime} :: {levelname:^8s} :: {name} | {message}",
        datefmt="%Y-%m-%d @ %H:%M:%S",
        style="{",
    )

    handlers = [
        logging.StreamHandler(stream=sys.stderr),
    ]
    if log_file is not None:
        handlers.append(logging.FileHandler(filename=log_file))

    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    sys.excepthook = handle_uncaught_exception


def get_var(var_name, default=None, *, check_not_set=True, var_type=str):
    """Read an environment variable.

    Parameters
    ----------
    var_name : str
        The name of the environment variable.
    default : object or None
        The default value of the variable.
    check_not_set : bool, optional
        If the value of the variable is `None`, which is the
        case when the variable is not set, then a `ValueError`
        is raised.
    var_type : callable, optional
        The type of the variable. Before returning the variable will
        be cast to this type.

    Returns
    -------
    var : str or None
        The value of the environment variable.
    """
    var = os.getenv(var_name, default)
    if check_not_set and var is None:
        raise ValueError(f"The variable ${var_name} must be set")

    return var_type(var)


def run_server(app_factory, name, argv=None):
    """Run a server app from the command line.

    This starts Flask's development web server. For development
    purposes only. For production use the corresponding docker file.

    Parameters
    ----------
    app_factory : callable
        A factory function that returns an instance of a flask app.
    name : str
        The server name. This will be printed in the help message.
    argv : list_like of str
        The command line arguments.
    """
    from dotenv import load_dotenv

    # Parse arguments
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description=f"Start the {name} server.",
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
    app = app_factory()
    app.run(host=args.host, port=args.port, threaded=True, debug=True)

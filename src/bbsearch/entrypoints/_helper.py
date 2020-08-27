"""Helper functions for server entry points."""
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
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="{asctime} :: {levelname:^8s} :: {name} | {message}",
        style="{",
        datefmt='%Y-%M-%d @ %H:%M:%S',
    )
    sys.excepthook = handle_uncaught_exception


def get_var(var_name, default=None, *, check_not_set=True):
    """Read an environment variable.

    Parameters
    ----------
    var_name : str
        The name of the environment variable.
    default : str or None
        The default value of the variable.
    check_not_set : bool
        If the value of the variable is `None`, which is the
        case when the variable is not set, then a `ValueError`
        is raised.

    Returns
    -------
    var : str or None
        The value of the environment variable.
    """
    var = os.getenv(var_name, default)
    if check_not_set and var is None:
        raise ValueError(f"The variable ${var_name} must be set")

    return var

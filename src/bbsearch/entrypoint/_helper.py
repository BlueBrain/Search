"""Helper functions for server entry points."""
import argparse
import collections
import logging
import os
import sys
import textwrap


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


class CombinedHelpFormatter(argparse.HelpFormatter):
    """Argparse formatter with raw text and default value display.

    This is a combination of `argparse.RawTextHelpFormatter` and
    `argparse.ArgumentDefaultsHelpFormatter`, and the implementation is
    almost literally copied from the `argparse` module.

    """

    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        help = textwrap.dedent(action.help).strip()
        if "%(default)" not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += "\n(default: %(default)s)"
        return help


def parse_args_or_environment(parser, env_variable_names, argv=None):
    from dotenv import load_dotenv

    # Parse CLI arguments
    cli_args = vars(parser.parse_args(argv))

    # Parse environment
    load_dotenv()
    environment_args = {}
    for arg_name, value_name in env_variable_names.items():
        value = os.environ.get(value_name)
        if value is not None:
            environment_args[arg_name] = value

    # Combine CLI and environment variables
    args = collections.ChainMap(cli_args, environment_args)

    # Check if all arguments were supplied
    for arg_name in env_variable_names:
        if arg_name not in args:
            parser.print_usage()
            parser.exit(
                status=1,
                message=f"The following arguments are required: --{arg_name.replace('_', '-')}\n",
            )

    return args

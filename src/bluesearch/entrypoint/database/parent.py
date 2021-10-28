"""Module implementing the high level CLI logic."""
import argparse
import logging
import sys
from typing import Optional, Sequence

from bluesearch.entrypoint.database import (
    add,
    convert_pdf,
    init,
    parse
)

def _setup_logging(logging_level: int):
    root_logger = logging.getLogger()

    root_logger.setLevel(logging_level)

    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CLI.

    This is the main entrypoint that defines different commands
    using subparsers.
    """
    parser = argparse.ArgumentParser(
        description="Database management utilities"
    )
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        help="Controls the verbosity",
        action="count",
        default=0,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Initialize subparsers
    add_parser = subparsers.add_parser(
        "add",
        help="Add parsed files to the database.",
        parents=[parent_parser],
    )
    add.init_parser(add_parser)

    convert_pdf_parser = subparsers.add_parser(
        "convert-pdf",
        help="Convert a PDF file to a TEI XML file.",
        parents=[parent_parser],
    )
    convert_pdf.init_parser(convert_pdf_parser)

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a database.",
        parents=[parent_parser],
    )
    init.init_parser(init_parser)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse raw files.",
        parents=[parent_parser],
    )
    parse.init_parser(parse_parser)


    command_map = {
        "add": add.run,
        "convert-pdf": convert_pdf.run,
        "init": init.run,
        "parse": parse.run,
    }

    # Do parsing
    args = parser.parse_args(argv)

    kwargs = vars(args)
    command = kwargs.pop("command")
    verbose = min(kwargs.pop("verbose"), 2)

    # Setup logging
    logging_level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    _setup_logging(logging_level_map[verbose])


    # Run logic
    return command_map[command](**kwargs)  # type: ignore

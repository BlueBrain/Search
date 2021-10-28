"""Module implementing the high level CLI logic."""
import argparse
import logging
import sys
from typing import Optional, Sequence

from bluesearch.entrypoint.database import (
        add,
        convert_pdf,
        init,
)
from bluesearch.entrypoint.database.parse import get_parser as get_parser_parse
from bluesearch.entrypoint.database.parse import run as run_parse


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
        action="store_true",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Initialize subparsers
    parser_parse = get_parser_parse()

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
        help="Initialize a database.""",
        parents=[parent_parser],
    )
    init.init_parser(init_parser)

    subparsers.add_parser(
        "parse",
        description=parser_parse.description,
        help=parser_parse.description,
        parents=[parser_parse, parent_parser],
        add_help=False,
    )

    command_map = {
        "add": add.run,
        "convert-pdf": convert_pdf.run,
        "init": init.run,
        "parse": run_parse,
    }

    # Do parsing
    args = parser.parse_args(argv)

    kwargs = vars(args)
    command = kwargs.pop("command")
    verbose = kwargs.pop("verbose")

    # Setup logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Run logic
    return command_map[command](**kwargs)  # type: ignore

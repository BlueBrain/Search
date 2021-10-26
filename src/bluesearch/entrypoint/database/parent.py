"""Module implementing the high level CLI logic."""
import argparse
from typing import Optional, Sequence

from bluesearch.entrypoint.database import convert_pdf
from bluesearch.entrypoint.database.add import get_parser as get_parser_add
from bluesearch.entrypoint.database.add import run as run_add
from bluesearch.entrypoint.database.init import get_parser as get_parser_init
from bluesearch.entrypoint.database.init import run as run_init
from bluesearch.entrypoint.database.parse import get_parser as get_parser_parse
from bluesearch.entrypoint.database.parse import run as run_parse


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CLI.

    This is the main entrypoint that defines different commands
    using subparsers.
    """
    parent_parser = argparse.ArgumentParser(description="Database management utilities")

    subparsers = parent_parser.add_subparsers(dest="command", required=True)

    # Initialize subparsers
    parser_add = get_parser_add()
    parser_init = get_parser_init()
    parser_parse = get_parser_parse()

    subparsers.add_parser(
        "add",
        description=parser_add.description,
        help=parser_add.description,
        parents=[parser_add],
        add_help=False,
    )
    subparsers.add_parser(
        "init",
        description=parser_init.description,
        help=parser_init.description,
        parents=[parser_init],
        add_help=False,
    )
    subparsers.add_parser(
        "parse",
        description=parser_parse.description,
        help=parser_parse.description,
        parents=[parser_parse],
        add_help=False,
    )
    convert_pdf_parser = subparsers.add_parser(
        "convert-pdf",
        help="Convert a PDF file to a TEI XML file.",
    )
    convert_pdf.init_parser(convert_pdf_parser)

    command_map = {
        "add": run_add,
        "convert-pdf": convert_pdf.run,
        "init": run_init,
        "parse": run_parse,
    }

    # Do parsing
    args = parent_parser.parse_args(argv)

    kwargs = vars(args)
    command = kwargs.pop("command")

    # Run logic
    return command_map[command](**kwargs)  # type: ignore

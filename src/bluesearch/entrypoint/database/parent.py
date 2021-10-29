"""Module implementing the high level CLI logic."""
import argparse
import logging
import sys
from collections import namedtuple
from typing import Optional, Sequence

from bluesearch.entrypoint.database import add, convert_pdf, init, parse

Cmd = namedtuple("Cmd", ["name", "help", "init_parser", "run"])


def _setup_logging(logging_level: int) -> None:
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
    parser = argparse.ArgumentParser(description="Database management utilities")
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

    # Define all commands, order matters (--help)
    cmds = [
        Cmd(
            name="add",
            help="Add parsed files to the database.",
            init_parser=add.init_parser,
            run=add.run,
        ),
        Cmd(
            name="convert-pdf",
            help="Convert a PDF file to a TEI XML file.",
            init_parser=convert_pdf.init_parser,
            run=convert_pdf.run,
        ),
        Cmd(
            name="init",
            help="Initialize a database.",
            init_parser=init.init_parser,
            run=init.run,
        ),
        Cmd(
            name="parse",
            help="Parse raw files.",
            init_parser=parse.init_parser,
            run=parse.run,
        ),
    ]

    # Initialize subparsers
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in cmds:
        cmd.init_parser(
            subparsers.add_parser(
                cmd.name,
                help=cmd.help,
                parents=[parent_parser],
            )
        )

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
    chosen_cmd = [cmd for cmd in cmds if cmd.name == command][0]

    return chosen_cmd.run(**kwargs)

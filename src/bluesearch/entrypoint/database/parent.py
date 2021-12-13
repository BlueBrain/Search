"""Module implementing the high level CLI logic."""
import argparse
import logging
import sys
from collections import namedtuple
from typing import Optional, Sequence

from bluesearch.entrypoint.database import add, convert_pdf, download, init, parse, topic_extract

Cmd = namedtuple("Cmd", ["help", "init_parser", "run"])


def _setup_logging(logging_level: int) -> None:
    root_logger = logging.getLogger()

    # Logging level
    root_logger.setLevel(logging_level)

    # Formatter
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    # Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CLI.

    This is the main entrypoint that defines different commands
    using subparsers.
    """
    parser = argparse.ArgumentParser(description="Database management utilities")

    # Define all commands, order matters (--help)
    cmds = {
        "add": Cmd(
            help="Add parsed files to the database.",
            init_parser=add.init_parser,
            run=add.run,
        ),
        "convert-pdf": Cmd(
            help="Convert a PDF file to a TEI XML file.",
            init_parser=convert_pdf.init_parser,
            run=convert_pdf.run,
        ),
        "download": Cmd(
            help="Download articles from different sources.",
            init_parser=download.init_parser,
            run=download.run,
        ),
        "init": Cmd(
            help="Initialize a database.",
            init_parser=init.init_parser,
            run=init.run,
        ),
        "parse": Cmd(
            help="Parse raw files.",
            init_parser=parse.init_parser,
            run=parse.run,
        ),
        "topic-extract": Cmd(
            help="Extract topic of article(s).",
            init_parser=topic_extract.init_parser,
            run=topic_extract.run,
        ),
    }

    # Create a verbosity parser (it will be a parent of all subparsers)
    verbosity_parser = argparse.ArgumentParser(
        add_help=False,
    )
    verbosity_parser.add_argument(
        "-v",
        "--verbose",
        help=(
            "Controls the verbosity by setting the logging level. "
            "Default: WARNING, -v: INFO, -vv: DEBUG"
        ),
        action="count",
        default=0,
    )

    # Initialize subparsers
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd_name, cmd in cmds.items():
        cmd.init_parser(
            subparsers.add_parser(
                cmd_name,
                help=cmd.help,
                parents=[verbosity_parser],
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
    return cmds[command].run(**kwargs)

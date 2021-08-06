"""Module implementing the high level CLI logic."""
import argparse
import sys
from typing import Optional, Sequence

from .add import get_parser as get_parser_add
from .add import run as run_add
from .init import get_parser as get_parser_init
from .init import run as run_init


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

    _ = subparsers.add_parser(
        "add",
        description=parser_add.description,
        help=parser_add.description,
        parents=[parser_add],
        add_help=False,
    )
    _ = subparsers.add_parser(
        "init",
        description=parser_init.description,
        help=parser_init.description,
        parents=[parser_init],
        add_help=False,
    )

    command_map = {
        "add": run_add,
        "init": run_init,
    }

    # Do parsing
    args = parent_parser.parse_args(argv)

    kwargs = vars(args)
    command = kwargs.pop("command")

    # Run logic
    command_map[command](**kwargs)  # type: ignore

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: nocover

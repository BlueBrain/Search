import argparse
import sys
from typing import List, Optional

from .add import get_parser as get_parser_add, run as run_add
from .init import get_parser as get_parser_init, run as run_init

def main(argv: Optional[List[str]] = None) -> int:
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

    # Do parsing
    args = parent_parser.parse_args(argv)

    print(args)
    kwargs = vars(args)
    command = kwargs.pop("command")

    # Run logic
    if command == "add":
        run_add(**kwargs)
    elif command == "init":
        run_init(**kwargs)

    return 0

if __name__ == "__main__":
    sys.exit(main())


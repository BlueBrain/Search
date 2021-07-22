import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parent_parser = argparse.ArgumentParser(description="Database management utilities")

    subparsers = parent_parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(
        "add",
        help="Add entries.",
        description="Add entries.",
    )
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize.",
        description="Initialize.",
    )

    args = parent_parser.parse_args()

    print(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())


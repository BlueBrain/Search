import argparse
import sys
from typing import List, Optional


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add entries.",
    )
    return parser


def run() -> None:
    print("Inside of the add command") 


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(**vars(args))


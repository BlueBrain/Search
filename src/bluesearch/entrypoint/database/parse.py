# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Parsing articles."""
import argparse
import json
import pickle  # nosec
from pathlib import Path
from typing import Iterable

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    PubmedXMLParser,
)


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Parse one or several articles.",
    )
    parser.add_argument(
        "article_type",
        type=str,
        choices=("cord19-json", "pmc-xml"),
        help="""
        Article format. If parsing several articles, all articles
        must have same format.
        """,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="""
        Path to a file or directory. If a directory, all articles
        inside the directory will be parsed.
        """,
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="""
        Path to a directory where parsed article(s) will be saved.
        If it does not exist yet, a directory with this path is created.
        """,
    )
    return parser


def run(
    *,
    article_type: str,
    input_path: Path,
    output_path: Path,
) -> None:
    """Parse one or several articles.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    inputs: Iterable[Path]
    if input_path.is_file():
        inputs = [input_path]
    elif input_path.is_dir():
        inputs = input_path.glob("*")
    else:
        raise ValueError(
            "Argument 'input_path' should be a path to an existing file or directory!"
        )

    for inp in inputs:
        parser_inst: ArticleParser
        if article_type == "cord19-json":
            with inp.open("r") as f_inp:
                parser_inst = CORD19ArticleParser(json.load(f_inp))
        elif article_type == "pmc-xml":
            parser_inst = PubmedXMLParser(inp)
        else:
            raise ValueError(f"Unsupported article type {article_type}")

        article = Article.parse(parser_inst)

        output_path.mkdir(exist_ok=True)
        # If we used .with_suffix(), then we would get file.pdf.xml --> file.pkl
        out = output_path / (inp.stem + ".pkl")

        with out.open("wb") as f_out:
            pickle.dump(article, f_out)

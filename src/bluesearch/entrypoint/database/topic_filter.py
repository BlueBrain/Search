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
"""Filter articles with relevant topics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the topic-filter subcommand.

    Parameters
    ----------
    parser
        The argument parser to initialise.

    Returns
    -------
    argparse.ArgumentParser
        The initialised argument parser. The same object as the `parser`
        argument.
    """
    parser.description = "Filter articles with relevant topics"

    parser.add_argument(
        "extracted_topics",
        type=Path,
        help="""
        Path to a .JSONL file that was an output of the `topic-extract`
        command.
        """,
    )
    parser.add_argument(
        "filter_config",
        type=Path,
        help="""
        Path to a .YAML file that defines what topics are relevant.
        """,
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="""
        Path to a .CSV file where rows are different articles
        and columns contain relevant information about these articles.
        """,
    )

    return parser

def validate(value: Any, key: str | None = None, level: int = -1) -> None:
    """Validate if section names of the config file are correct."""

    allowed_section_names = [
        {"accept", "reject"},
        {"article", "journal"},
        {"pubmed", "arxiv", "pmc", "biorxiv", "medrxiv"}
    ]

    n_mandatory_levels = len(allowed_section_names)

    if level == -1:
        # The entire config file has been provided
        pass

    elif 0 <= level < n_mandatory_levels:
        # We are in a level that we want to validate
        if key not in allowed_section_names[level]:
            raise ValueError(f"Illegal section named {key}")

    elif level >= n_mandatory_levels:
        # We are in a level that we don't want to validate
        return

    else:
        raise ValueError(f"Illegal level {level}")

    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            validate(subvalue, subkey, level + 1)

def run(
    *,
    extracted_topics: Path,
    filter_config: Path,
    output_file: Path,
) -> int:
    """Filter articles containing relevant topics.

    Parameter description and potential defaults are documented inside of the
    `init_parser` function.
    """
    import pprint
    import yaml

    import pandas as pd

    from bluesearch.database.article import ArticleSource
    from bluesearch.utils import JSONL


    # Create pattern list
    with filter_config.open() as f_config:
        config = yaml.safe_load(f_config)
    
    pprint.pprint(config)

    # Validation
    validate(config)

    # For each (article | journal, source) tuple you should have a set of 
    # patterns (order does not matter) that we will try to match

    topics = JSONL.load_jsonl(extracted_topics)

    output_columns = [
        "path",
        "element_in_file",
        "accept",
    ]

    df = pd.DataFrame(columns=output_columns)

    df.to_csv(output_file, index=False)


    return 0

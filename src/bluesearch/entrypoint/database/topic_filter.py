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
        Path to a .JSONL file that defines all the rules for filtering.
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


def run(
    extracted_topics: Path,
    filter_config: Path,
    output_file: Path,
) -> int:
    """Filter articles containing relevant topics.

    Parameter description and potential defaults are documented inside of the
    `init_parser` function.
    """
    import pandas as pd

    from bluesearch.database.topic_info import TopicInfo
    from bluesearch.database.topic_rule import TopicRule, check_accepted
    from bluesearch.utils import JSONL

    # Create pattern list
    config = JSONL.load_jsonl(filter_config)

    # Extract rules
    topic_rules_accept, topic_rules_reject = [], []
    for raw_rule in config:
        rule = TopicRule(
            level=raw_rule.get("level"),
            source=raw_rule.get("source"),
            pattern=raw_rule.get("pattern"),
        )
        label = raw_rule["label"]

        if label == "accept":
            topic_rules_accept.append(rule)
        elif label == "reject":
            topic_rules_reject.append(rule)
        else:
            ValueError(f"Unsupported label {label}")

    # Populate
    output_rows = []  # If True we accept that give topic info

    for topic_info_raw in JSONL.load_jsonl(extracted_topics):
        topic_info = TopicInfo.from_dict(topic_info_raw)
        output_rows.append(
            {
                "path": topic_info.path,
                "element_in_file": topic_info.element_in_file,
                "accept": check_accepted(
                    topic_info, topic_rules_accept, topic_rules_reject
                ),
                "source": topic_info.source,
            }
        )

    df = pd.DataFrame(output_rows)
    df.to_csv(output_file, index=False)

    return 0

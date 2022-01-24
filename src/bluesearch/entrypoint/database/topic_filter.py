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
import re
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bluesearch.database.topic_info import ArticleSource

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

@dataclass
class TopicRule:
    # None always represent wildcards
    level: str | None = None  # "article" or "journal"
    source: str | ArticleSource | None = None  # "arxiv", ... , "pubmed"
    pattern: str | re.Pattern | None = None  # regex pattern to match

    def __post_init__(self) -> None:
        """Validate inputs."""
        if self.level is not None and self.level not in {"article", "journal"}:
            raise ValueError(f"Unsupported level {self.level}")

        if self.pattern is not None:
            try:
                self.pattern = re.compile(self.pattern)
            except re.error:
                raise ValueError(f"Unsupported pattern {self.pattern}") from None

        if self.source is not None:
            try:
                self.source = ArticleSource(self.source)
            except ValueError:
                raise ValueError(f"Unsupported source {self.source}") from None

    def match(self, topic_info: TopicInfo) -> bool:
        """Determine whether a topic_info matches the rule."""
        # Source
        if self.source is not None and self.source is not topic_info.source:
            return False

        if self.pattern is None:
            return True

        if self.level is None or self.level == "article":
            for topic_list in topic_info.article_topics.values():
                if any(self.pattern.search(topic) for topic in topic_list):
                    return True

        if self.level is None or self.level == "journal":
            for topic_list in topic_info.journal_topics.values():
                if any(self.pattern.search(topic) for topic in topic_list):
                    return True

        return False

def run(
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
    from bluesearch.database.topic_info import TopicInfo
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

    # Extract infos
    topic_infos = [TopicInfo.from_dict(d) for d in JSONL.load_jsonl(extracted_topics)]

    # Populate
    decisions = []

    for topic_info in topic_infos:
        # Go through rejection rules
        rejected = False
        for topic_rule in topic_rules_reject:
            rule_satisfied = topic_rule.match(topic_info)
            if rule_satisifed:
                rejected = True
                break

        if rejected:
            decisions.append(False)
            continue

        
        # Go through acceptance rules
        accepted = False
        for topic_rule in topic_rules_accept:
            rule_satisfied = topic_rule.match(topic_info)
            if rule_satisfied:
                accepted = True
                break

        decisions.append(accepted)

    output_rows = []  # If True we accept that give topic info
    for topic_info, decision in zip(topic_infos, decisions):
        output_rows.append(
            {
                "path": topic_info.path,
                "element_in_file": topic_info.element_in_file,
                "accept": decision,
                "source": topic_info.source,
            }
        )

    df = pd.DataFrame(output_rows)
    df.to_csv(output_file, index=False)

    return 0

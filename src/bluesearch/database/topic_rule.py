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
"""Implementation of the TopicRule data structure."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo


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


def check_accepted(
        topic_info: TopicInfo,
        topic_rules_accept: Iterable[TopicRule],
        topic_rules_reject: Iterable[TopicRule],
) -> bool:
    """Check whether the rules are satisfied.

    The `topic_info` needs to satisfy both of the below
    conditions to be accepted:
      * At least one rule within `topic_rules_accept` is satisfied
      * No rules in `topic_rules_reject` are satisfied
    """
    # Go through rejection rules
    for topic_rule in topic_rules_reject:
        if topic_rule.match(topic_info):
            return False

    # Go through acceptance rules
    for topic_rule in topic_rules_accept:
        if topic_rule.match(topic_info):
            return True

    return False

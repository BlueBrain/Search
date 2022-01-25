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
from typing import Iterable, Optional

import pydantic

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo


class TopicRule(pydantic.BaseModel, arbitrary_types_allowed=True):
    """The topic rule."""

    # None always represent wildcards
    level: Optional[str] = None  # "article" or "journal"
    source: Optional[ArticleSource] = None  # "arxiv", ... , "pubmed"
    pattern: Optional[re.Pattern] = None  # regex pattern to match

    @pydantic.validator("level")
    def check_level_value(cls, value):
        """Check the level parameter value."""
        if value is not None and value not in {"article", "journal"}:
            raise ValueError(f"Unsupported level {value}")
        return value

    @pydantic.validator("source", pre=True)
    def convert_source_to_article_source(cls, value) -> ArticleSource | None:
        """Check the source parameter value."""
        if value is not None:
            try:
                value = ArticleSource(value)
            except ValueError:
                raise ValueError(f"Unsupported source {value}") from None

        return value

    @pydantic.validator("pattern", pre=True)
    def convert_pattern_to_re(cls, value) -> re.Pattern | None:
        """Check the pattern parameter value."""
        if value is not None:
            try:
                value = re.compile(value)
            except re.error:
                raise ValueError(f"Unsupported pattern {value}") from None

        return value

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

    Parameters
    ----------
    topic_info
        Topic info to accept or reject.
    topic_rules_accept
        List of topic rules to accept a given topic_info.
    topic_rules_reject
        List of topic rules to reject a given topic_info.

    Returns
    -------
    bool
        If True, the topic info matches satisfies both conditions explained above.
        If False, at least one of the conditions is not satisfied.
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

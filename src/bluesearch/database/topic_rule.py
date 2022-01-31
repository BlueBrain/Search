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
from typing import Any, Iterable

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo


class TopicRule:
    """Rule for accepting/rejecting an article based on topic matching criteria.

    Parameters
    ----------
    level
        Level of the topic information to match, must be "article" or "journal".
        Passing `None` will match any level.
    source
        Article source, must be a valid ArticleSource (e.g. "arxiv", "pmc", ...).
        Passing `None` will match any source.
    pattern
        Regular expression for matching the topic names of a given article.
        Passing `None` will match the name of any topic.
    """

    def __init__(
        self,
        level: str | None = None,
        source: str | ArticleSource | None = None,
        pattern: str | re.Pattern | None = None,
    ):
        if level is not None and level not in {"article", "journal"}:
            raise ValueError(f"Unsupported level {level}.")
        self.level = level
        self.source = ArticleSource(source) if source is not None else None
        self.pattern = re.compile(pattern) if pattern is not None else None

    def match(self, topic_info: TopicInfo) -> bool:
        """Determine whether a topic_info matches the rule.

        Note that the keys (topic sources) of the `topic_info.article_topics`
        and `topic_info.journal_topics` dictionaries are completely disregarded.
        And all the values (lists) are simply concatenated.

        """
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

    def __eq__(self, other: Any) -> bool:
        """Compare to another topic rule."""
        if not isinstance(other, TopicRule):
            return False
        return (
            self.level == other.level
            and self.source == other.source
            and self.pattern == other.pattern
        )


def check_topic_rules(
    topic_info: TopicInfo,
    topic_rules_accept: Iterable[TopicRule],
    topic_rules_reject: Iterable[TopicRule],
) -> bool:
    """Check whether the topic info of an article satisfies given topic rules."""

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

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
"""Implementation of the TopicInfo data structure."""
from __future__ import annotations

import copy
import datetime
import pathlib
from dataclasses import dataclass, field
from typing import Any

import bluesearch
from bluesearch.database.article import ArticleSource


@dataclass
class TopicInfo:
    """The topic information extracted from a journal article.

    For the spec see the following GitHub issue/comment:
    https://github.com/BlueBrain/Search/issues/518#issuecomment-985525160
    """

    source: ArticleSource
    path: str | pathlib.Path
    element_in_file: int | None = None
    article_topics: dict[str, list[str]] = field(init=False, default_factory=dict)
    journal_topics: dict[str, list[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Run the post-initialization."""
        self.creation_date = datetime.datetime.now()
        self.path = pathlib.Path(self.path).resolve()

    @staticmethod
    def _add_topics(
        mapping: dict[str, list[str]], kind: str, topics: list[str]
    ) -> None:
        """Add topics to a mapping with collection of topics.

        Parameters
        ----------
        mapping
            A mapping of the form kind -> list-of-topics that shall be
            updated in-place. For example ``{"MeSH": ["topic 1", "topic 2"]}``.
        kind
            The topic kind. Corresponds to a key in ``mapping``.
        topics
            The topics to add. Corresponds to a value in ``mapping``.
        """
        updated_topics = mapping.get(kind, []) + topics
        mapping[kind] = sorted(set(updated_topics))

    def add_article_topics(self, kind: str, topics: list[str]) -> None:
        """Add article topics.

        Parameters
        ----------
        kind
            The topic kind. For example "MeSH" or "MAG".
        topics
            A list of the topics to add.
        """
        self._add_topics(self.article_topics, kind, topics)

    def add_journal_topics(self, kind: str, topics: list[str]) -> None:
        """Add journal topics.

        Parameters
        ----------
        kind
            The topic kind. For example "MeSH" or "MAG".
        topics
            A list of the topics to add.
        """
        self._add_topics(self.journal_topics, kind, topics)

    def json(self) -> dict:
        """Convert the contents of this class to a structured dictionary.

        Apart from the source, path and topic entries a "metadata" top-level
        key will be added containing a dictionary with entries "created-date"
        and "bbs-version".

        Returns
        -------
        dict
            The structure dictionary with all topic information.
        """
        metadata: dict[str, Any] = {
            "created-date": self.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
            "bbs-version": bluesearch.__version__,
        }
        if self.element_in_file is not None:
            metadata["element_in_file"] = self.element_in_file

        json = {
            "source": self.source.value,
            "path": str(self.path),
            "topics": {
                "article": copy.deepcopy(self.article_topics),
                "journal": copy.deepcopy(self.journal_topics),
            },
            "metadata": metadata,
        }

        return json

    @classmethod
    def from_dict(cls, data: dict) -> TopicInfo:
        """Parse topic info from a dictionary.

        Parameters
        ----------
        data
            The dictionary to parse

        Returns
        -------
        TopicInfo
            The parsed topic info.
        """
        source = ArticleSource(data["source"])
        path = data["path"]
        element_in_file = data["metadata"].get("element_in_file")
        topic_info = cls(source, path, element_in_file)
        for topic_type, topics in data["topics"]["article"].items():
            topic_info.add_article_topics(topic_type, topics)
        for topic_type, topics in data["topics"]["journal"].items():
            topic_info.add_journal_topics(topic_type, topics)

        return topic_info

#  Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
#  Copyright (C) 2022 Blue Brain Project, EPFL.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import json
import logging
import pathlib
from typing import Generator, Sequence

logger = logging.getLogger(__name__)


class MeSHTree:
    def __init__(self, tree_number_to_label: dict[str, str]) -> None:
        self.tree_number_to_label = tree_number_to_label
        self.label_to_tree_numbers = {}
        for tree_number, label in tree_number_to_label.items():
            if label not in self.label_to_tree_numbers:
                self.label_to_tree_numbers[label] = []
            self.label_to_tree_numbers[label].append(tree_number)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> MeSHTree:
        with open(path) as fh:
            tree_number_to_label = json.load(fh)
        return cls(tree_number_to_label)

    @staticmethod
    def parents(tree_number: str) -> Generator[str, None, None]:
        parts = tree_number.split(".")
        for n in reversed(range(1, len(parts))):
            yield ".".join(parts[:n])

    def parent_topics(self, topic: str) -> list[str]:
        parent_topics = set()
        for tree_number in self.label_to_tree_numbers[topic]:
            for parent in self.parents(tree_number):
                parent_topics.add(self.tree_number_to_label[parent])

        return sorted(parent_topics)


def resolve_parents(topic_list: Sequence[str], mesh_tree: MeSHTree) -> list[str]:
    resolved = []
    for topic in topic_list:
        resolved.append(topic)
        resolved.extend(mesh_tree.parent_topics(topic))

    return resolved

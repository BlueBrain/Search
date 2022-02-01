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
"""Utilities for handling MeSH topic data."""
from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterable
from typing import Generator

logger = logging.getLogger(__name__)


class MeSHTree:
    """The hierarchical tree of MeSH topics.

    The MeSH topic ontology forms a tree with most general topics at the
    root and the most specific topics as the leafs. Here's a part of a MeSH
    topic hierarchy:

        Natural Science Disciplines [H01]
            Biological Science Disciplines [H01.158]
                Biology [H01.158.273]
                    Botany [H01.158.273.118]
                        Ethnobotany [H01.158.273.118.299]
                        Pharmacognosy [H01.158.273.118.598]
                            Herbal Medicine [H01.158.273.118.598.500]
                    Cell Biology [H01.158.273.160]
        ...

    The full data can be found in the NLM's MeSH browser under
    https://meshb.nlm.nih.gov/.

    The topics are uniquely identified by their tree number (e.g. `H01.158`),
    while the same topic label can appear in different places.

    Parameters
    ----------
    tree_number_to_label
        The MeSH tree data. This dictionary should have tree numbers
        (e.g. `H01.158.273`) as keys, and topic labels (e.g. `Biology`)
        as values.
    """

    def __init__(self, tree_number_to_label: dict[str, str]) -> None:
        self.tree_number_to_label = tree_number_to_label
        self.label_to_tree_numbers: dict[str, list[str]] = {}
        for tree_number, label in tree_number_to_label.items():
            if label not in self.label_to_tree_numbers:
                self.label_to_tree_numbers[label] = []
            self.label_to_tree_numbers[label].append(tree_number)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> MeSHTree:
        """Initialise the MeSH tree from a JSON file.

        Parameters
        ----------
        path
            The path to the JSON file containing the MeSH tree data. See
            the `tree_number_to_label` parameter of the `MeSHTree`
            constructor for the data specification.

        Returns
        -------
        MeSHTree
            An initialised instance of the MeSHTree.
        """
        with open(path) as fh:
            tree_number_to_label = json.load(fh)
        return cls(tree_number_to_label)

    @staticmethod
    def parents(tree_number: str) -> Generator[str, None, None]:
        """Generate all parent tree numbers.

        For example, given the tree number `H01.158.273` the parent tree
        numbers are `H01.158` and `H01`.

        Parameters
        ----------
        tree_number
            A MeSH tree number, e.g. `H01.158.273`.

        Yields
        ------
        The tree numbers of all parents of the given tree number.
        """
        parts = tree_number.split(".")
        for n in reversed(range(1, len(parts))):
            yield ".".join(parts[:n])

    def parent_topics(self, topic: str) -> set[str]:
        """Find all parent topic labels of a given topic.

        Note that a topic label does not have to be unique and can be
        assigned to multiple tree numbers. This method resolves all
        parent topics from all tree numbers that have the given label.

        Parameters
        ----------
        topic
            A MeSH topic label.

        Returns
        -------
        list
            All parent topic labels.
        """
        parent_topics = set()
        for tree_number in self.label_to_tree_numbers[topic]:
            for parent in self.parents(tree_number):
                parent_topics.add(self.tree_number_to_label[parent])

        return parent_topics


def resolve_parents(topics: Iterable[str], mesh_tree: MeSHTree) -> set[str]:
    """Enhance the topic list by parents of all given topics.

    Parameters
    ----------
    topics
        A collection of MeSH topics.
    mesh_tree
        An instance of `MeSHTree`.

    Returns
    -------
    set[str]
        A set with the input topics and all their parent topics.
    """
    resolved = set(topics)
    for topic in topics:
        resolved |= mesh_tree.parent_topics(topic)

    return resolved

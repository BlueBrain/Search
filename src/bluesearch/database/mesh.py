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

import collections
import json
import logging
import pathlib
import re
from collections.abc import Iterable
from typing import Generator, TextIO

logger = logging.getLogger(__name__)


class MeSHTree:
    """The hierarchical tree of MeSH topics.

    The MeSH topic ontology forms a tree with most general topics at the
    root and the most specific topics as the leafs. Here's a part of a MeSH
    topic hierarchy

    .. code-block:: text

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


def parse_tree_numbers(nt_stream: TextIO) -> dict[str, str]:
    """Parse the MeSH topic tree from a stream of MeSH RDF N-tuples.

    Parameters
    ----------
    nt_stream
        A text stream of MeSH RDF N-tuples. This is intended to work with
        the content of the MeSH files downloaded from the following website:
        https://nlmpubs.nlm.nih.gov/projects/mesh/rdf

    Returns
    -------
    dict[str, str]
        A dictionary representing the parsed MeSH topic tree. The keys are
        the tree numbers that uniquely identify a topic. The values are the
        corresponding topic labels. Note that the topic labels are not
        unique. For example, the two tree numbers `F04.096.628.255.500` and
        `H01.158.610.030` have both the same label "Cognitive Neuroscience".
    """
    id_to_label: dict[str, str] = {}
    id_to_tree_numbers = collections.defaultdict(list)

    # Regexes we need for parsing
    # Each line must be a triple subject, predicate, object.
    p_line = re.compile(r"(<.*>) (<.*>) (.*) \.")
    # We're only interested in subjects that represent descriptors. It appears
    # their ID is of the form "Dxxx..." where "xxx" are digits.
    p_desc = re.compile(r"<http://id\.nlm\.nih\.gov/mesh/\d{4}/(D\d{3,})>")
    # The topic label is in quotes and is followed by a language suffix. We'll
    # only keep labels that are in English
    p_en_label = re.compile(r"\"(.*)\"@en")
    # The "\d{4}" part is going to be the year, e.g. "2022", the actual tree
    # number is some combination of characters that we leave open.
    p_tree_number = re.compile(r"<http://id\.nlm\.nih\.gov/mesh/\d{4}/(.*)>")

    # The two predicates we'll be looking for
    pred_label = "<http://www.w3.org/2000/01/rdf-schema#label>"
    pred_tree_number = "<http://id.nlm.nih.gov/mesh/vocab#treeNumber>"

    for i, line in enumerate(nt_stream):
        if i % 1_000_000 == 0:
            logger.info(f"Parsed {i:,d} lines")

        # Parse the triple
        m_line = p_line.fullmatch(line.strip())
        if not m_line:
            raise RuntimeError(f"The line is not a valid triple: {line!r}")
        subj, pred, obj = m_line.groups()

        # Extract the descriptor ID
        m_desc = p_desc.fullmatch(subj)
        if not m_desc:
            # Subject is not a descriptor
            continue
        id_ = m_desc.group(1)

        # Parse the descriptor label or tree number
        if pred == pred_label:
            m_label = p_en_label.fullmatch(obj)
            if not m_label:
                continue  # not an English label
            label = m_label.group(1)
            if id_ in id_to_label:
                raise RuntimeError(
                    f"Multiple labels for ID={id_}: {id_to_label[id_]}, {label}"
                )
            id_to_label[id_] = label
        elif pred == pred_tree_number:
            m_tree_number = p_tree_number.fullmatch(obj)
            if not m_tree_number:
                raise RuntimeError(f"Cannot parse tree number: {obj}")
            id_to_tree_numbers[id_].append(m_tree_number.group(1))

    # Given "id => label" and "id => tree numbers" find "tree number => label"
    logger.info("Labeling tree numbers")
    tree_number_to_label = {}
    for id_, label in id_to_label.items():
        for tree_number in id_to_tree_numbers[id_]:
            if tree_number in tree_number_to_label:
                raise RuntimeError(f"Duplicate tree number: {tree_number}")
            tree_number_to_label[tree_number] = label

    return tree_number_to_label

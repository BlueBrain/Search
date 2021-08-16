"""Tests covering the handling of identifiers."""

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
from typing import List

import pandas as pd

from bluesearch.database.identifiers import generate_uuids

# Should be assigned different UUIDs.
DATA_DIFFERENT_2 = [
    # No shared values.
    ("a_1", "b_1"),
    ("a_2", "b_2"),
    # No shared values (NAs, right).
    ("a_3", None),
    ("a_4", None),
    # No shared values (NAs, left).
    (None, "b_3"),
    (None, "b_4"),
    # A conflicting value (right).
    ("a_5", "b_5"),
    ("a_5", "b_6"),
    # A conflicting value (right, with NA).
    ("a_6", "b_7"),
    ("a_6", "b_8"),
    ("a_6", None),
    # A conflicting value (left).
    ("a_7", "b_9"),
    ("a_8", "b_9"),
    # A conflicting value (left, with NA).
    ("a_9", "b_10"),
    ("a_10", "b_10"),
    (None, "b_10"),
]

# Should be assigned different UUIDs.
DATA_DIFFERENT_3 = [
    ("a_1", "b_1", "c_1"),
    ("a_1", "b_2", None),
    ("a_2", "b_3", "c_2"),
    ("a_3", "b_3", None),
]

# Should be assigned same UUID.
DATA_SAME = [
    # All values shared.
    ("a_1", "b_1"),
    ("a_1", "b_1"),
    # No conflicting value (after, right).
    ("a_2", "b_2"),
    ("a_2", None),
    # No conflicting value (after, left).
    ("a_3", "b_3"),
    (None, "b_3"),
    # No conflicting value (before, right).
    ("a_4", None),
    ("a_4", "b_4"),
    # No conflicting value (before, left).
    (None, "b_5"),
    ("a_5", "b_5"),
]

IDENTIFIERS_2 = ["id_1", "id_2"]
IDENTIFIERS_3 = [*IDENTIFIERS_2, "id_3"]


def check(result: pd.DataFrame, expected: List) -> None:
    """Check if the resulting cluster are as expected."""
    indices = result.groupby("cluster_uuid", sort=False).groups.values()
    clusters = list(map(lambda x: x.to_list(), indices))
    assert clusters == expected


class TestClustering:
    def test_generate_uuids_different(self):
        metadata = pd.DataFrame(DATA_DIFFERENT_2, columns=IDENTIFIERS_2)
        result = generate_uuids(metadata, IDENTIFIERS_2)
        count = len(result)
        expected = [[i] for i in range(count)]
        check(result, expected)

    def test_generate_uuids_same(self):
        metadata = pd.DataFrame(DATA_SAME, columns=IDENTIFIERS_2)
        result = generate_uuids(metadata, IDENTIFIERS_2)
        count = len(result)
        expected = [[i, j] for i, j in zip(range(0, count, 2), range(1, count, 2))]
        check(result, expected)

    def test_generate_uuids_empty(self):
        metadata = pd.DataFrame(DATA_DIFFERENT_3, columns=IDENTIFIERS_3)
        result = generate_uuids(metadata, IDENTIFIERS_3)
        count = len(result)
        expected = [[i] for i in range(count)]
        check(result, expected)

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

from typing import List, Union

import pandas as pd
import pytest

from bluesearch.database.identifiers import generate_uuids

# Should be assigned different UUIDs.
ALL_DIFFERENT = [
    # No shared values.
    (0, "a_1", "b_1"),
    (1, "a_2", "b_2"),
    # No shared values (NAs, right).
    (2, "a_3", None),
    (3, "a_4", None),
    # No shared values (NAs, left).
    (4, None, "b_3"),
    (5, None, "b_4"),
    # A conflicting value (right).
    (6, "a_5", "b_5"),
    (7, "a_5", "b_6"),
    # A conflicting value (right, with NA).
    (8, "a_6", "b_7"),
    (9, "a_6", "b_8"),
    (10, "a_6", None),
    # A conflicting value (left).
    (11, "a_7", "b_9"),
    (12, "a_8", "b_9"),
    # A conflicting value (left, with NA).
    (13, "a_9", "b_10"),
    (14, "a_10", "b_10"),
    (15, None, "b_10"),
]

# Should be assigned different UUIDs.
EMPTY_CLUSTERS = [
    (0, "a_1", "b_1", "c_1"),
    (1, "a_1", "b_2", None),
    (2, "a_2", "b_3", "c_2"),
    (3, "a_3", "b_3", None),
]

# Should be assigned same UUID per pair.
IDENTICAL_PAIRS = [
    # All values shared.
    (0, "a_1", "b_1"),
    (0, "a_1", "b_1"),
    # No conflicting value (after, right).
    (1, "a_2", "b_2"),
    (1, "a_2", None),
    # No conflicting value (after, left).
    (2, "a_3", "b_3"),
    (2, None, "b_3"),
    # No conflicting value (before, right).
    (3, "a_4", None),
    (3, "a_4", "b_4"),
    # No conflicting value (before, left).
    (4, None, "b_5"),
    (4, "a_5", "b_5"),
]


def clusters(df: pd.DataFrame, column: Union[str, int]) -> List[List[int]]:
    """Return the clusters according to the given column."""
    indices = df.groupby(column, sort=False).groups.values()
    return list(map(lambda x: x.to_list(), indices))


class TestClustering:
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(ALL_DIFFERENT, id="all_different"),
            pytest.param(EMPTY_CLUSTERS, id="empty_clusters"),
            pytest.param(IDENTICAL_PAIRS, id="identical_pairs"),
        ],
    )
    def test_generate_uuids(self, data):
        metadata = pd.DataFrame(data)
        expected = clusters(metadata, 0)
        result = generate_uuids(metadata, metadata.columns[1:])
        found = clusters(result, "cluster_uuid")
        assert found == expected

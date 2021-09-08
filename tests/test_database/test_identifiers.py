"""Tests covering the clustering of identifiers."""

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

from bluesearch.database.identifiers import generate_uids

# Should be assigned different UIDs.

DIFFERENT_BASIC = [
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

DIFFERENT_ORDER = [
    # A conflicting value (right, with NA, before).
    (0, "a_1", None),
    (1, "a_1", "b_1"),
    (2, "a_1", "b_2"),
    # A conflicting value (right, with NA, after).
    (3, "a_2", "b_3"),
    (4, "a_2", "b_4"),
    (5, "a_2", None),
    # A conflicting value (left, with NA, before).
    (6, None, "b_5"),
    (7, "a_3", "b_5"),
    (8, "a_4", "b_5"),
    # A conflicting value (left, with NA, after).
    (9, "a_5", "b_6"),
    (10, "a_6", "b_6"),
    (11, None, "b_6"),
]

DIFFERENT_COMPLEX = [
    # A conflicting value (left and NA middle).
    (0, "a_1", None, "c_1"),
    (1, "a_2", "b_1", "c_1"),
    # A conflicting value (left and NA right).
    (2, "a_3", "b_2", None),
    (3, "a_4", "b_2", "c_2"),
    # A conflicting value (middle and NA left).
    (4, None, "b_3", "c_3"),
    (5, "a_5", "b_4", "c_3"),
    # A conflicting value (middle and NA right).
    (6, "a_6", "b_5", None),
    (7, "a_6", "b_6", "c_4"),
    # No conflicting value (middle and right).
    (8, "a_7", None, "c_5"),
    (9, "a_7", "b_7", None),
    # No conflicting value (left and right).
    (10, None, "b_8", "c_6"),
    (11, "a_8", "b_8", None),
    # No conflicting value (left and middle).
    (12, None, "b_9", "c_7"),
    (13, "a_9", None, "c_7"),
]

# Should be assigned same UID per pair.

IDENTICAL = [
    # All values shared.
    (0, "a_1", "b_1"),
    (0, "a_1", "b_1"),
    # Shared values (NAs, right).
    (1, "a_2", None),
    (1, "a_2", None),
    # Shared values (NAs, left).
    (2, None, "b_2"),
    (2, None, "b_2"),
]


def clusters(df: pd.DataFrame, column: Union[str, int]) -> List[List[int]]:
    """Return the clusters according to the given column."""
    indices = df.groupby(column, sort=False).groups.values()
    return list(map(lambda x: x.to_list(), indices))


class TestClustering:
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(DIFFERENT_BASIC, id="different_basic_cases"),
            pytest.param(DIFFERENT_ORDER, id="different_processing_order"),
            pytest.param(DIFFERENT_COMPLEX, id="different_complex_cases"),
            pytest.param(IDENTICAL, id="identical_cases"),
        ],
    )
    def test_generate_uids(self, data):
        metadata = pd.DataFrame(data)
        expected = clusters(metadata, 0)
        result = generate_uids(metadata, metadata.columns[1:])
        found = clusters(result, "uid")
        assert found == expected

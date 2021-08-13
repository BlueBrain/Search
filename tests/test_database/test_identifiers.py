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

import pandas as pd

from bluesearch.database.identifiers import generate_uuids

DATA = [
    # Should be assigned different UUIDs.
    # - No shared values.
    ("a_1", "b_1"),
    ("a_2", "b_2"),
    # - No shared values (NAs, right).
    ("a_3", None),
    ("a_4", None),
    # - No shared values (NAs, left).
    (None, "b_3"),
    (None, "b_4"),
    # - A conflicting value (right).
    ("a_5", "b_5"),
    ("a_5", "b_6"),
    # - A conflicting value (left).
    ("a_6", "b_7"),
    ("a_7", "b_7"),
    # Should be assigned same UUID.
    # - All values shared.
    ("a_8", "b_8"),
    ("a_8", "b_8"),
    # - No conflicting value (after, right).
    ("a_9", "b_9"),
    ("a_9", None),
    # - No conflicting value (after, left).
    ("a_10", "b_10"),
    (None, "b_10"),
    # - No conflicting value (before, right).
    ("a_11", None),
    ("a_11", "b_11"),
    # - No conflicting value (before, left).
    (None, "b_12"),
    ("a_12", "b_12"),
]
IDENTIFIERS = ["id_1", "id_2"]


class TestClustering:
    def test_generate_uuids(self):
        metadata = pd.DataFrame(DATA, columns=IDENTIFIERS)
        results = generate_uuids(metadata, IDENTIFIERS)

        indices = results.groupby("cluster_uuid", sort=False).groups.values()
        clusters = list(map(lambda x: x.to_list(), indices))

        different = [[i] for i in range(10)]
        same = [[i, j] for i, j in zip(range(10, 19, 2), range(11, 20, 2))]
        expected = different + same

        assert clusters == expected

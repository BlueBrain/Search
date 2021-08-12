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


class TestClustering:
    def test_generate_uuids(self):
        data = [
            # Should be assigned different UUIDs.
            (None, "a"),
            (None, "b"),
            ("1", None),
            ("2", None),
            ("1", "a"),
            ("1", "b"),
            ("2", "a"),
            ("2", "b"),
            # Should be assigned same UUID.
            (None, "c"),
            ("3", "c"),
            # Should be assigned same UUID.
            ("4", None),
            ("4", "d"),
        ]
        identifiers = ["id_1", "id_2"]
        metadata = pd.DataFrame(data, columns=identifiers)

        results = generate_uuids(metadata, identifiers)

        indices = results.groupby("cluster_uuid", sort=False).groups.values()
        clusters = list(map(lambda x: x.to_list(), indices))
        expected = [[0], [1], [2], [3], [4], [5], [6], [7], [8, 9], [10, 11]]
        assert clusters == expected

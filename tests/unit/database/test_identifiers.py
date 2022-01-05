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

from __future__ import annotations

import pytest

from bluesearch.database.identifiers import generate_uid


class TestIdentifiers:
    @pytest.mark.parametrize(
        "identifiers, expected",
        [
            pytest.param(
                ("a", "b"), "aca14e654bc28ce1c1e8131004244d64", id="all-defined"
            ),
            pytest.param(
                ("b", "a"), "82ca240c4a3f5579a5c33404af58e41b", id="all-defined-reverse"
            ),
            pytest.param(
                ("a", None), "4b515f920fbbc7954fc5a68bb746b109", id="with-none"
            ),
            pytest.param(
                (None, "a"), "77f283f2e87b852ed7a881e6f638aa80", id="with-none-reverse"
            ),
            pytest.param((None, None), None, id="all-none"),
            pytest.param(
                (None, 0), "14536e026b2a39caf27f3da802e7fed6", id="none-and-zero"
            ),
        ],
    )
    def test_generate_uid(self, identifiers, expected):
        # By running this test several times and on different platforms during the CI,
        # this test checks that 'generate_uid(...)' is deterministic across platforms
        # and Python processes.

        if expected is None:
            with pytest.raises(ValueError):
                generate_uid(identifiers)

        else:
            result = generate_uid(identifiers)
            assert result == expected

            # Check determinism.
            result_bis = generate_uid(identifiers)
            assert result == result_bis

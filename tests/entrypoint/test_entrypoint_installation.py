"""Tests covering entrypoint installation."""

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

import subprocess

import pytest


@pytest.mark.parametrize(
    "entrypoint_name",
    [
        "compute_embeddings",
        "create_database",
        "create_mining_cache",
        "embedding_server",
        "mining_server",
        "search_server",
    ],
)
def test_entrypoint(entrypoint_name):
    subprocess.check_call([entrypoint_name, "--help"])

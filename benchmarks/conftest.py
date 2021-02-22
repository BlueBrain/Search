"""Configuration of pytest benchmarks."""

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

import pytest


def pytest_addoption(parser):
    parser.addoption("--embedding_server", default="", help="Embedding server URI")
    parser.addoption("--mining_server", default="", help="Mining server URI")
    parser.addoption("--mysql_server", default="", help="MySQL server URI")
    parser.addoption("--search_server", default="", help="Search server URI")


@pytest.fixture(scope="session")
def benchmark_parameters(request):
    return {
        "embedding_server": request.config.getoption("--embedding_server"),
        "mining_server": request.config.getoption("--mining_server"),
        "mysql_server": request.config.getoption("--mysql_server"),
        "search_server": request.config.getoption("--search_server"),
    }

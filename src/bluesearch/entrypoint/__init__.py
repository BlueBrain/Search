"""Subpackage containing all the entry points."""

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

from .compute_embeddings import run_compute_embeddings
from .create_database import run_create_database
from .create_mining_cache import run_create_mining_cache
from .embedding_server import get_embedding_app, run_embedding_server
from .mining_server import get_mining_app, run_mining_server
from .search_server import get_search_app, run_search_server

__all__ = [
    "get_embedding_app",
    "get_mining_app",
    "get_search_app",
    "run_compute_embeddings",
    "run_create_mining_cache",
    "run_create_database",
    "run_embedding_server",
    "run_mining_server",
    "run_search_server",
]

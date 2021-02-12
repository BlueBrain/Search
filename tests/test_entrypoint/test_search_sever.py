"""Collection of tests focused on "search_server" entrypoint."""

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

import pathlib
from unittest.mock import Mock

import numpy as np
import pytest

from bluesearch.entrypoint import get_search_app
from bluesearch.server.search_server import SearchServer


@pytest.mark.parametrize(
    "embeddings_path,models,models_path",
    [
        ("path_1", ["A", "B", "C"], "path_a"),
        ("path_2", ["X", "Y"], "path_b"),
    ],
)
def test_send_through(tmpdir, monkeypatch, embeddings_path, models, models_path):
    tmpdir = pathlib.Path(str(tmpdir))
    logfile = tmpdir / "log.txt"

    monkeypatch.setenv("BBS_SEARCH_LOG_FILE", str(logfile))
    monkeypatch.setenv("BBS_SEARCH_MODELS_PATH", models_path)
    monkeypatch.setenv("BBS_SEARCH_EMBEDDINGS_PATH", embeddings_path)
    monkeypatch.setenv("BBS_SEARCH_MODELS", ",".join(models))
    monkeypatch.setenv("BBS_SEARCH_DB_URL", "some_url")
    monkeypatch.setenv("BBS_SEARCH_MYSQL_USER", "some_user")
    monkeypatch.setenv("BBS_SEARCH_MYSQL_PASSWORD", "some_pwd")

    fake_sqlalchemy = Mock()
    fake_H5 = Mock()
    fake_H5.find_populated_rows.return_value = np.arange(1, 11)
    fake_search_server_inst = Mock(spec=SearchServer)
    fake_search_server_class = Mock(return_value=fake_search_server_inst)

    monkeypatch.setattr(
        "bluesearch.entrypoint.search_server.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr("bluesearch.utils.H5", fake_H5)
    monkeypatch.setattr(
        "bluesearch.server.search_server.SearchServer", fake_search_server_class
    )

    server_app = get_search_app()

    # Checks
    fake_search_server_class.assert_called_once()
    fake_H5.find_populated_rows.assert_called_once()
    fake_sqlalchemy.create_engine.assert_called_once()

    assert server_app is fake_search_server_inst

    args, kwargs = fake_search_server_class.call_args

    assert args[0] == pathlib.Path(models_path)
    assert args[1] == pathlib.Path(embeddings_path)
    np.testing.assert_array_equal(args[2], np.arange(1, 11))
    assert args[3] is fake_sqlalchemy.create_engine.return_value
    assert args[4] == models

"""Collection of tests focusing on the `embedding_server` entrypoint."""

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

from bluesearch.entrypoint import get_embedding_app
from bluesearch.server.embedding_server import EmbeddingServer


def test_environment_reading(monkeypatch, tmpdir):
    tmpdir = pathlib.Path(str(tmpdir))
    logfile = tmpdir / "log.txt"
    logfile.touch()

    fake_embedding_server_inst = Mock(spec=EmbeddingServer)
    fake_embedding_server_class = Mock(return_value=fake_embedding_server_inst)

    monkeypatch.setattr(
        "bluesearch.server.embedding_server.EmbeddingServer",
        fake_embedding_server_class,
    )

    # Mock all of our embedding models
    embedding_models = ["SentTransformer"]

    for model in embedding_models:
        monkeypatch.setattr(f"bluesearch.embedding_models.{model}", Mock())

    monkeypatch.setenv("BBS_EMBEDDING_LOG_FILE", str(logfile))

    embedding_app = get_embedding_app()

    assert embedding_app is fake_embedding_server_inst

    args, _ = fake_embedding_server_class.call_args

    assert len(args) == 1
    assert isinstance(args[0], dict)

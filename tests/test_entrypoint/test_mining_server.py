"""Collection of tests focused on the `mining_server`."""

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

import pytest

from bluesearch.entrypoint import get_mining_app


@pytest.mark.parametrize(
    ("db_type", "sqlite_db_exists"),
    (
        ("sqlite", True),
        ("sqlite", False),
        ("mysql", False),
        ("wrong", False),
    ),
)
def test_send_through(
    tmpdir, monkeypatch, db_type, sqlite_db_exists, entity_types, spacy_model_path
):
    tmpdir = pathlib.Path(str(tmpdir))
    logfile = tmpdir / "log.txt"
    db_path = tmpdir / "something.db"

    if sqlite_db_exists:
        db_path.parent.mkdir(exist_ok=True, parents=True)
        db_path.touch()

    monkeypatch.setenv("BBS_MINING_LOG_FILE", str(logfile))
    monkeypatch.setenv("BBS_MINING_DB_TYPE", db_type)
    monkeypatch.setenv("BBS_MINING_DB_URL", str(db_path))
    monkeypatch.setenv("BBS_MINING_MYSQL_USER", "some_user")
    monkeypatch.setenv("BBS_MINING_MYSQL_PASSWORD", "some_pwd")
    monkeypatch.setenv("BBS_DATA_AND_MODELS_DIR", str(spacy_model_path))

    fake_sqlalchemy = Mock()
    fake_mining_server_inst = Mock()
    fake_mining_server_class = Mock(return_value=fake_mining_server_inst)

    monkeypatch.setattr(
        "bluesearch.server.mining_server.MiningServer", fake_mining_server_class
    )
    monkeypatch.setattr(
        "bluesearch.entrypoint.mining_server.sqlalchemy", fake_sqlalchemy
    )

    if db_type not in {"mysql", "sqlite"}:
        with pytest.raises(ValueError):
            get_mining_app()
    else:
        mining_app = get_mining_app()

        fake_mining_server_class.assert_called_once()
        assert mining_app == fake_mining_server_inst

        args, kwargs = fake_mining_server_class.call_args
        assert not args
        assert kwargs["connection"] == fake_sqlalchemy.create_engine.return_value
        assert "ee" in kwargs["models_libs"]
        assert isinstance(kwargs["models_libs"]["ee"], dict)
        assert len(kwargs["models_libs"]["ee"]) == len(entity_types)

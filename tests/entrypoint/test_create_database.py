"""Testing the create_database entrypoint."""

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

from bluesearch.entrypoint.create_database import run_create_database


@pytest.mark.parametrize(
    (
        "cord_data_path",
        "db_type",
        "db_url",
        "sqlite_exists",
        "log_file",
        "only_mark_bad_sentences",
    ),
    [
        (
            "data_1",
            "mysql",
            "my_server.ch/my_database",
            False,
            "folder_1/a.log",
            True,
        ),
        ("data_2", "sqlite", "database.db", False, "folder_2/b.log", False),
        ("data_3", "sqlite", "database.db", True, "folder_2/b.log", False),
        ("data_4", "wrong", "no_database_here", False, "folder_3/c.log", False),
    ],
)
def test_send_through(
    monkeypatch,
    tmpdir,
    cord_data_path,
    db_type,
    db_url,
    sqlite_exists,
    log_file,
    only_mark_bad_sentences,
):
    # Preparations
    tmpdir = pathlib.Path(str(tmpdir))
    log_file = tmpdir / log_file
    log_file.parent.mkdir()

    # Patching
    fake_getpass = Mock()
    fake_getpass.getpass.return_value = "whatever"
    fake_sqlalchemy = Mock()
    fake_database_creation = Mock()
    fake_mark_bad_sentences = Mock()

    if db_type == "sqlite":
        db_url = tmpdir / db_url
        if sqlite_exists:
            db_url.touch()

    monkeypatch.setattr("builtins.input", Mock())
    monkeypatch.setattr("bluesearch.entrypoint.create_database.getpass", fake_getpass)
    monkeypatch.setattr(
        "bluesearch.entrypoint.create_database.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr(
        "bluesearch.database.CORD19DatabaseCreation", fake_database_creation
    )
    monkeypatch.setattr(
        "bluesearch.database.mark_bad_sentences", fake_mark_bad_sentences
    )

    argv = [
        f"--cord-data-path={cord_data_path}",
        f"--db-type={db_type}",
        f"--db-url={db_url}",
        f"--log-file={log_file}",
    ]

    if only_mark_bad_sentences:
        argv.append("--only-mark-bad-sentences")

    if db_type in {"mysql", "sqlite"}:
        run_create_database(argv)
    else:
        with pytest.raises(SystemExit):
            run_create_database(argv)

        return

    # Checks
    if only_mark_bad_sentences:
        fake_database_creation.assert_not_called()

    else:
        fake_database_creation.assert_called_once()

        args, kwargs = fake_database_creation.call_args

        assert kwargs["data_path"] == cord_data_path
        assert kwargs["engine"] == fake_sqlalchemy.create_engine.return_value

    fake_mark_bad_sentences.assert_called_once()

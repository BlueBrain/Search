"""Testing the create_database entrypoint."""
import pathlib
from unittest.mock import Mock

import pytest

from bbsearch.entrypoint.create_database import run_create_database


@pytest.mark.parametrize(
    (
        "data_path",
        "db_type",
        "database_url",
        "sqlite_exists",
        "log_dir",
        "log_name",
        "only_mark_bad_sentences",
    ),
    [
        (
            "data_1",
            "mysql",
            "my_server.ch/my_database",
            False,
            "folder_1",
            "a.log",
            True,
        ),
        ("data_2", "sqlite", "database.db", False, "folder_2", "b.log", False),
        ("data_3", "sqlite", "database.db", True, "folder_2", "b.log", False),
        ("data_4", "wrong", "no_database_here", False, "folder_3", "c.log", False),
    ],
)
def test_send_through(
    monkeypatch,
    tmpdir,
    data_path,
    db_type,
    database_url,
    sqlite_exists,
    log_dir,
    log_name,
    only_mark_bad_sentences,
):
    # Preparations
    tmpdir = pathlib.Path(str(tmpdir))
    log_dir = tmpdir / log_dir
    log_dir.mkdir()

    # Patching
    fake_getpass = Mock()
    fake_getpass.getpass.return_value = "whatever"
    fake_sqlalchemy = Mock()
    fake_database_creation = Mock()
    fake_mark_bad_sentences = Mock()

    if db_type == "sqlite":
        database_url = tmpdir / database_url
        if sqlite_exists:
            database_url.touch()

    monkeypatch.setattr("builtins.input", Mock())
    monkeypatch.setattr("bbsearch.entrypoint.create_database.getpass", fake_getpass)
    monkeypatch.setattr(
        "bbsearch.entrypoint.create_database.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr(
        "bbsearch.database.CORD19DatabaseCreation", fake_database_creation
    )
    monkeypatch.setattr("bbsearch.database.mark_bad_sentences", fake_mark_bad_sentences)

    argv = [
        f"--data-path={data_path}",
        f"--db-type={db_type}",
        f"--database-url={database_url}",
        f"--log-dir={log_dir}",
        f"--log-name={log_name}",
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

        assert kwargs["data_path"] == data_path
        assert kwargs["engine"] == fake_sqlalchemy.create_engine.return_value

    fake_mark_bad_sentences.assert_called_once()

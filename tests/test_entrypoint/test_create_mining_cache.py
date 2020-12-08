import pytest
from unittest.mock import Mock

from bbsearch.entrypoint import run_create_mining_cache


def test_help(capsys):
    with pytest.raises(SystemExit) as error:
        run_create_mining_cache(["--help"])
    stdout, stderr = capsys.readouterr()

    assert error.value.code == 0
    assert stdout.startswith("usage:")
    assert stderr == ""


@pytest.mark.parametrize(
    ("db_type", "database_url"),
    (
        ("mysql", "my_url"),
        ("sqlite", "my_url"),
    ),
)
def test_send_through(monkeypatch, db_type, database_url):
    # Monkey-patching
    fake_pandas = Mock()
    fake_sqlalchemy = Mock()
    fake_create_mining_cache = Mock()
    monkeypatch.setattr("bbsearch.entrypoint.create_mining_cache.pd", fake_pandas)
    monkeypatch.setattr("bbsearch.entrypoint.create_mining_cache.sqlalchemy", fake_sqlalchemy)
    monkeypatch.setattr("bbsearch.database.mining_cache.CreateMiningCache", fake_create_mining_cache)

    # Construct arguments
    argv = [
        f"--db-type={db_type}",
        f"--database-url={database_url}",
    ]

    # Call entrypoint method
    # run_create_mining_cache(argv)

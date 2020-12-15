"""Collection of tests focused on the `mining_server`."""
import pathlib
from unittest.mock import Mock

import pandas as pd
import pytest

from bbsearch.entrypoint import get_mining_app


@pytest.mark.parametrize("db_type", ["sqlite", "mysql", "wrong"])
def test_send_through(tmpdir, monkeypatch, db_type):
    tmpdir = pathlib.Path(str(tmpdir))
    logfile = tmpdir / "log.txt"
    db_path = tmpdir / "something.db"

    monkeypatch.setenv("BBS_MINING_LOG_FILE", str(logfile))
    monkeypatch.setenv("BBS_MINING_DB_TYPE", db_type)
    monkeypatch.setenv("BBS_MINING_SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("BBS_MINING_MYSQL_URL", "something.db")
    monkeypatch.setenv("BBS_MINING_MYSQL_USER", "some_user")
    monkeypatch.setenv("BBS_MINING_MYSQL_PASSWORD", "some_pwd")

    fake_sqlalchemy = Mock()
    fake_dvc = Mock()
    fake_dvc.load_ee_models_library.return_value = pd.DataFrame()
    fake_mining_server_inst = Mock()
    fake_mining_server_class = Mock(return_value=fake_mining_server_inst)

    monkeypatch.setattr(
        "bbsearch.server.mining_server.MiningServer", fake_mining_server_class
    )
    monkeypatch.setattr("bbsearch.entrypoint.mining_server.sqlalchemy", fake_sqlalchemy)
    monkeypatch.setattr("bbsearch.entrypoint.mining_server.DVC", fake_dvc)

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
        assert isinstance(kwargs["models_libs"]["ee"], pathlib.Path)

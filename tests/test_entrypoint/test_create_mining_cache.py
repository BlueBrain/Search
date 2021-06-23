"""Tests covering the creation of the mining cache database."""

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

from bluesearch.entrypoint import run_create_mining_cache


def test_help(capsys):
    with pytest.raises(SystemExit) as error:
        run_create_mining_cache(["--help"])
    stdout, stderr = capsys.readouterr()

    assert error.value.code == 0
    assert stdout.startswith("usage:")
    assert stderr == ""


def test_missing_sqlite_db():
    with pytest.raises(FileNotFoundError, match="^No database found"):
        run_create_mining_cache(
            [
                "--data-and-models-dir",
                "/some/path",
                "--db-type",
                "sqlite",
                "--db-url",
                "fake$?#",
            ],
        )


@pytest.mark.parametrize(
    (
        "db_type",
        "db_url",
        "target_table_name",
        "n_processes_per_model",
        "restrict_to_etypes",
    ),
    (
        (
            "mysql",
            "my_url",
            "mysql_cache_table",
            4,
            "CHEMICAL,ORGANISM",
        ),
        (
            "sqlite",
            "my_url",
            "sqlite_cache_table",
            12,
            "CHEMICAL,invalid",
        ),
    ),
)
def test_send_through(
    monkeypatch,
    tmpdir,
    db_type,
    db_url,
    target_table_name,
    n_processes_per_model,
    restrict_to_etypes,
    entity_types,
    spacy_model_path,
):
    # Monkey-patching
    fake_sqlalchemy = Mock()
    fake_create_mining_cache = Mock()
    monkeypatch.setattr(
        "bluesearch.entrypoint.create_mining_cache.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr(
        "bluesearch.database.CreateMiningCache", fake_create_mining_cache
    )
    monkeypatch.setattr(
        "bluesearch.database.CreateMiningCache", fake_create_mining_cache
    )
    monkeypatch.setattr(
        "bluesearch.entrypoint.create_mining_cache.getpass.getpass",
        lambda _: "fake_password",
    )

    # Create temporary sqlite database
    if db_type == "sqlite":
        db_url = pathlib.Path(tmpdir) / "my.db"
        db_url.touch()

    # Construct arguments
    argv = [
        f"--data-and-models-dir={spacy_model_path}",
        f"--db-type={db_type}",
        f"--db-url={db_url}",
        f"--target-table-name={target_table_name}",
        f"--n-processes-per-model={n_processes_per_model}",
        f"--restrict-to-etypes={restrict_to_etypes}",
    ]

    # Call entrypoint method
    # import pdb; pdb.set_trace()
    run_create_mining_cache(argv)

    # Checks
    # Check that CreateMiningCache(...) was called once and get its arguments
    fake_create_mining_cache.assert_called_once()
    args, kwargs = fake_create_mining_cache.call_args

    # Construct the restricted etypes
    available_models = set(restrict_to_etypes.split(",")) & set(entity_types)

    # Check the args/kwargs
    assert kwargs["database_engine"] == fake_sqlalchemy.create_engine()
    assert isinstance(kwargs["ee_models_paths"], dict)
    assert len(kwargs["ee_models_paths"]) == len(available_models)
    assert kwargs["target_table_name"] == target_table_name
    assert kwargs["workers_per_model"] == n_processes_per_model

    # Check that CreateMiningCache.construct() was called
    fake_create_mining_cache().construct.assert_called_once()

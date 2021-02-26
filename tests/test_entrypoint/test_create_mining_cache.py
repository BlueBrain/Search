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

import pandas as pd
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
        "restrict_to_models",
    ),
    (
        (
            "mysql",
            "my_url",
            "mysql_cache_table",
            4,
            "path/to/model_1,path/to/model_2",
        ),
        (
            "sqlite",
            "my_url",
            "sqlite_cache_table",
            12,
            "path/to/model_3,invalid",
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
    restrict_to_models,
):
    # Monkey-patching
    data_dir = "/data_dir"
    df_model_library = pd.DataFrame(
        columns=["entity_type", "entity_type_name", "model_id", "model_path"],
        data=[
            [
                "CELL_COMPARTMENT",
                "CELLULAR_COMPONENT",
                "path/to/model_1",
                f"{data_dir}/path/to/model_1",
            ],
            [
                "CELL_TYPE",
                "CELL_TYPE",
                "path/to/model_2",
                f"{data_dir}/path/to/model_2",
            ],
            ["CHEMICAL", "CHEBI", "path/to/model_3", f"{data_dir}/path/to/model_3"],
        ],
    )
    fake_load_ee_models_library = Mock()
    fake_load_ee_models_library.return_value = df_model_library
    fake_sqlalchemy = Mock()
    fake_create_mining_cache = Mock()
    monkeypatch.setattr(
        "bluesearch.entrypoint.create_mining_cache.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr(
        "bluesearch.entrypoint.create_mining_cache.load_ee_models_library",
        fake_load_ee_models_library,
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
        "--data-and-models-dir=/some/fake/path",
        f"--db-type={db_type}",
        f"--db-url={db_url}",
        f"--target-table-name={target_table_name}",
        f"--n-processes-per-model={n_processes_per_model}",
        f"--restrict-to-models={restrict_to_models}",
    ]

    # Call entrypoint method
    # import pdb; pdb.set_trace()
    run_create_mining_cache(argv)

    # Checks
    # Check that CreateMiningCache(...) was called once and get its arguments
    fake_create_mining_cache.assert_called_once()
    args, kwargs = fake_create_mining_cache.call_args

    # Construct the restricted model library data frame
    selected_models = restrict_to_models.split(",")
    df_model_library_selected = df_model_library[
        df_model_library["model_id"].isin(selected_models).tolist()
    ]

    # Check the args/kwargs
    assert kwargs["database_engine"] == fake_sqlalchemy.create_engine()
    assert isinstance(kwargs["ee_models_library"], pd.DataFrame)
    assert len(df_model_library_selected) > 0
    assert kwargs["ee_models_library"].equals(df_model_library_selected)
    assert kwargs["target_table_name"] == target_table_name
    assert kwargs["workers_per_model"] == n_processes_per_model

    # Check that CreateMiningCache.construct() was called
    fake_create_mining_cache().construct.assert_called_once()

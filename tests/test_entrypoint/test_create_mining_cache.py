import pathlib
from unittest.mock import Mock

import pandas as pd
import pytest

from bbsearch.entrypoint import run_create_mining_cache


def test_help(capsys):
    with pytest.raises(SystemExit) as error:
        run_create_mining_cache(["--help"])
    stdout, stderr = capsys.readouterr()

    assert error.value.code == 0
    assert stdout.startswith("usage:")
    assert stderr == ""


def test_invalid_db_type():
    with pytest.raises(ValueError, match="^Invalid database type"):
        run_create_mining_cache(["--db-type", "wrong-type"])


def test_missing_sqlite_db():
    with pytest.raises(FileNotFoundError, match="^No database found"):
        run_create_mining_cache(
            ["--db-type", "sqlite", "--database-url", "fake$?#"],
        )


@pytest.mark.parametrize(
    (
        "db_type",
        "database_url",
        "target_table_name",
        "ee_models_library_file",
        "n_processes_per_model",
        "restrict_to_models",
    ),
    (
        (
            "mysql",
            "my_url",
            "mysql_cache_table",
            "models_1.csv",
            4,
            "/path/to/model_1,/path/to/model_2",
        ),
        (
            "sqlite",
            "my_url",
            "sqlite_cache_table",
            "models_2.csv",
            12,
            "/path/to/model_3,invalid",
        ),
    ),
)
def test_send_through(
    monkeypatch,
    tmpdir,
    db_type,
    database_url,
    target_table_name,
    ee_models_library_file,
    n_processes_per_model,
    restrict_to_models,
):
    # Monkey-patching
    df_model_library = pd.DataFrame(
        columns=["entity_type", "model", "entity_type_name"],
        data=[
            ["CELL_COMPARTMENT", "/path/to/model_1", "CELLULAR_COMPONENT"],
            ["CELL_TYPE", "/path/to/model_2", "CELL_TYPE"],
            ["CHEMICAL", "/path/to/model_3", "CHEBI"],
        ],
    )
    fake_pandas = Mock()
    fake_pandas.read_csv.return_value = df_model_library
    fake_sqlalchemy = Mock()
    fake_create_mining_cache = Mock()
    monkeypatch.setattr("bbsearch.entrypoint.create_mining_cache.pd", fake_pandas)
    monkeypatch.setattr(
        "bbsearch.entrypoint.create_mining_cache.sqlalchemy", fake_sqlalchemy
    )
    monkeypatch.setattr("bbsearch.database.CreateMiningCache", fake_create_mining_cache)
    monkeypatch.setattr(
        "bbsearch.entrypoint.create_mining_cache.getpass.getpass",
        lambda: "fake_password",
    )

    # Create temporary sqlite database
    if db_type == "sqlite":
        database_url = pathlib.Path(tmpdir) / "my.db"
        database_url.touch()

    # Construct arguments
    argv = [
        f"--db-type={db_type}",
        f"--database-url={database_url}",
        f"--target-table-name={target_table_name}",
        f"--ee-models-library-file={ee_models_library_file}",
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
        df_model_library["model"].isin(selected_models)
    ]

    # Check the args/kwargs
    assert kwargs["database_engine"] == fake_sqlalchemy.create_engine()
    assert isinstance(kwargs["ee_models_library"], pd.DataFrame)
    assert kwargs["ee_models_library"].equals(df_model_library_selected)
    assert kwargs["target_table_name"] == target_table_name
    assert kwargs["workers_per_model"] == n_processes_per_model

    # Check that CreateMiningCache.construct() was called
    fake_create_mining_cache().construct.assert_called_once()

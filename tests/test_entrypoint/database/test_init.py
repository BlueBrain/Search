import pathlib

import pytest

from bluesearch.entrypoint.database.parent import main

def test_mysql_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["init", "a", "--db-type=mysql"])

def test_sqlite(tmpdir):
    tmpdir = pathlib.Path(str(tmpdir))
    db_path = tmpdir / "database.db"

    args_and_opts = [
        "init",
        str(db_path),
        "--db-type=sqlite",
    ]

    assert not db_path.exists()

    main(args_and_opts)

    assert db_path.exists()

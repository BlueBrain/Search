import pathlib

import pytest
import sqlalchemy

from bluesearch.entrypoint.database.parent import main
from bluesearch.entrypoint.database.schemas import schema_articles, schema_sentences


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

    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    metadata = sqlalchemy.MetaData(engine)
    metadata.reflect(engine)
    tables = metadata.sorted_tables

    expected_metadata = sqlalchemy.MetaData()
    schema_articles(expected_metadata)
    schema_sentences(expected_metadata)
    expected_tables = expected_metadata.sorted_tables

    assert len(tables) == len(expected_tables)

    for table, expected in zip(tables, expected_tables):
        assert table.compare(expected)

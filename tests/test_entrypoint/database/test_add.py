import pathlib

import pytest
import sqlalchemy

from bluesearch.entrypoint.database.parent import main


@pytest.fixture()
def engine_sqlite(tmpdir):
    db_url = pathlib.Path(str(tmpdir)) / "database.db"
    eng = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    # Schema
    metadata = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "articles",
        metadata,
        sqlalchemy.Column(
            "article_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("title", sqlalchemy.Text()),
    )

    # Table
    with eng.begin() as connection:
        metadata.create_all(connection)

    return eng

def test_mysql_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["add", "a", "b", "c", "--db-type=mysql"])

def test_unknown_parser():
    with pytest.raises(ValueError, match="Unsupported parser"):
        main(["add", "dburl", "WrongParser", "path_to_files"])


def test_sqlite_cord19(engine_sqlite, jsons_path):
    # Create a dummy database
    path_jsons  = pathlib.Path(__file__).parent.parent.parent / "data" / "cord19_v35"
    all_paths = sorted(path_jsons.rglob("*.json"))

    n_articles = len(all_paths)

    for path in all_paths:
        args_and_opts = [
            "add",
            engine_sqlite.url.database,
            "CORD19ArticleParser",
            str(path),
            "--db-type=sqlite",
        ]

        main(args_and_opts)

    # Check
    with engine_sqlite.begin() as connection:
        query = """SELECT COUNT(*) FROM ARTICLES"""
        n_rows, = connection.execute(query).fetchone()

    assert n_rows == n_articles > 0


import pathlib
import pickle

import pytest
import sqlalchemy

from bluesearch.database.article import Article
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


def test_sqlite_cord19(engine_sqlite, tmpdir):
    input_folder = pathlib.Path(str(tmpdir))
    n_files = 3

    input_paths = [input_folder / f"{i}.pkl" for i in range(n_files)]

    for i, input_path in enumerate(input_paths):
        article = Article(
            title=f"title_{i}",
            authors=[f"author_{i}"],
            abstract=f"abstract_{i}",
            section_paragraphs=[("Conclusion", f"conclusion_{i}")],
        )
        with open(input_path, "wb") as f:
            pickle.dump(article, f)

        args_and_opts = [
            "add",
            engine_sqlite.url.database,
            str(input_path),
            "--db-type=sqlite",
        ]

        main(args_and_opts)

    # Check
    with engine_sqlite.begin() as connection:
        query = """SELECT COUNT(*) FROM ARTICLES"""
        (n_rows,) = connection.execute(query).fetchone()

    assert n_rows == n_files > 0

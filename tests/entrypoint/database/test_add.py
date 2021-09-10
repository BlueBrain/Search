import pathlib
import pickle

import pytest
import sqlalchemy

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main


def test_mysql_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["add", "a", "b", "--db-type=mysql"])


def test_sqlite_cord19(bbs_database_engine, tmpdir):
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
            bbs_database_engine.url.database,
            str(input_path),
            "--db-type=sqlite",
        ]

        main(args_and_opts)

    # Check
    with bbs_database_engine.begin() as connection:
        query = """SELECT COUNT(*) FROM ARTICLES"""
        (n_rows,) = connection.execute(query).fetchone()

    assert n_rows == n_files > 0

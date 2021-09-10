import pathlib
import pickle
from unittest.mock import MagicMock

import pytest
import sqlalchemy

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main


@pytest.mark.xfail
def test_mysql_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["add", "a", "b", "--db-type=mysql"])


@pytest.mark.parametrize("a", [1, 2])
def test_sqlite_cord19(bbs_database_session, tmpdir, monkeypatch, a):
    # Patching
    fake_create_engine = MagicMock()
    fake_create_engine().connect().__enter__.return_value = bbs_database_session

    monkeypatch.setattr("bluesearch.entrypoint.database.add.sqlalchemy.create_engine", fake_create_engine)

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
            "some-database-url",
            str(input_path),
            "--db-type=sqlite",
        ]
        main(args_and_opts)

    # Check
    query = """SELECT COUNT(*) FROM articles"""
    (n_rows,) = bbs_database_session.execute(query).fetchone()

    assert n_rows == n_files > 0

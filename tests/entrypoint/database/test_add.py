import pickle

import pytest
import sqlalchemy

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main
from bluesearch.entrypoint.database.schemas import schema_articles, schema_sentences


@pytest.fixture()
def engine_sqlite(tmp_path):
    db_url = tmp_path / "database.db"
    eng = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    # Schema
    metadata = sqlalchemy.MetaData()
    schema_articles(metadata)
    schema_sentences(metadata)

    # Table
    with eng.begin() as connection:
        metadata.create_all(connection)

    return eng


def test_mysql_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["add", "a", "b", "--db-type=mysql"])


def test_sqlite_cord19(engine_sqlite, tmp_path):
    path_to_pkl = tmp_path / "pkl_files"
    path_to_pkl.mkdir()
    n_files = 3

    pkl_files = [path_to_pkl / f"{i}.pkl" for i in range(n_files)]
    articles = [
        Article(
            title=f"title_{i}",
            authors=[f"author_{i}"],
            abstract=f"abstract_{i}",
            section_paragraphs=[("Conclusion", f"conclusion_{i}")],
        )
        for i in range(n_files)
    ]
    for article, pkl_file in zip(articles, pkl_files):
        with pkl_file.open("wb") as f:
            pickle.dump(article, f)

    query_articles = "SELECT COUNT(*) FROM articles"
    query_sentences = "SELECT COUNT(*) FROM sentences"

    # Test adding single article
    for pkl_file in pkl_files:
        args_and_opts = [
            "add",
            engine_sqlite.url.database,
            str(pkl_file),
            "--db-type=sqlite",
        ]
        main(args_and_opts)

    (n_rows_articles,) = engine_sqlite.execute(query_articles).fetchone()
    assert n_rows_articles == n_files

    (n_rows_sentences,) = engine_sqlite.execute(query_sentences).fetchone()
    assert n_rows_sentences > 0

    engine_sqlite.execute("DELETE FROM articles")

    (n_rows_articles,) = engine_sqlite.execute(query_articles).fetchone()
    assert n_rows_articles == 0

    # Test adding multiple articles
    args_and_opts = [
        "add",
        engine_sqlite.url.database,
        str(path_to_pkl),
        "--db-type=sqlite",
    ]
    main(args_and_opts)

    (n_rows_articles,) = engine_sqlite.execute(query_articles).fetchone()
    assert n_rows_articles == n_files

    (n_rows_sentences,) = engine_sqlite.execute(query_sentences).fetchone()
    assert n_rows_sentences > 0

    # Test adding something that does not exist
    with pytest.raises(ValueError):
        args_and_opts = [
            "add",
            engine_sqlite.url.database,
            str(path_to_pkl / "dir_that_does_not_exists"),
            "--db-type=sqlite",
        ]
        main(args_and_opts)

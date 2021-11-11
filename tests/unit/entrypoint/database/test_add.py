from unittest.mock import Mock

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


def test_sqlite_cord19(engine_sqlite, tmp_path, monkeypatch, model_entities):
    # Reuse a spacy model fixture
    monkeypatch.setattr(
        "bluesearch.utils.load_spacy_model", Mock(return_value=model_entities)
    )

    parsed_dir = tmp_path / "parsed_files"
    parsed_dir.mkdir()
    n_files = 3

    parsed_files = [parsed_dir / f"{i}.json" for i in range(n_files)]
    articles = [
        Article(
            title=f"title_{i}",
            authors=[f"author_{i}"],
            abstract=f"abstract_{i}",
            section_paragraphs=[("Conclusion", f"conclusion_{i}")],
            uid=f"uid_{i}",
            pubmed_id=f"pubmed_id_{i}",
            pmc_id=f"pmc_id_{i}",
            doi=f"doi_{i}",
        )
        for i in range(n_files)
    ]
    for article, parsed_file in zip(articles, parsed_files):
        serialized = article.to_json()
        parsed_file.write_text(serialized, "utf-8")

    query_articles = "SELECT COUNT(*) FROM articles"
    query_sentences = "SELECT COUNT(*) FROM sentences"

    # Test adding single article
    for parsed_file in parsed_files:
        args_and_opts = [
            "add",
            engine_sqlite.url.database,
            str(parsed_file),
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

    engine_sqlite.execute("DELETE FROM sentences")
    (n_rows_sentences,) = engine_sqlite.execute(query_sentences).fetchone()
    assert n_rows_sentences == 0

    # Test adding multiple articles
    args_and_opts = [
        "add",
        engine_sqlite.url.database,
        str(parsed_dir),
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
            str(parsed_dir / "dir_that_does_not_exists"),
            "--db-type=sqlite",
        ]
        main(args_and_opts)


def test_no_articles(tmp_path):
    parsed_dir = tmp_path / "empty"
    parsed_dir.mkdir()

    with pytest.raises(RuntimeWarning, match=r"No article was loaded from '.*'!"):
        main(["add", "test.db", str(parsed_dir)])


def test_no_sentences(tmp_path, engine_sqlite):
    parsed_dir = tmp_path / "parsed_files"
    parsed_dir.mkdir()

    article = Article(
        title="Title",
        authors=["Author"],
        abstract="Abstract",
        section_paragraphs=[],
        pubmed_id="PubMed ID",
        pmc_id="PMC ID",
        doi="DOI",
        uid="UID",
    )

    serialized = article.to_json()
    parsed_file = parsed_dir / "article.pkl"
    parsed_file.write_text(serialized, "utf-8")

    with pytest.raises(RuntimeWarning, match=r"No sentence was extracted from '.*'!"):
        main(["add", engine_sqlite.url.database, str(parsed_file)])

"""Adding an article to the database."""
import argparse


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Add entries to the database.",
    )
    parser.add_argument(
        "db_url",
        type=str,
        help="""
        The location of the database depending on the database type.

        For MySQL the server URL should be provided, for SQLite the
        location of the database file. Generally, the scheme part of
        the URL should be omitted, e.g. for MySQL the URL should be
        of the form 'my_sql_server.ch:1234/my_database' and for SQLite
        of the form '/path/to/the/local/database.db'.
        """,
    )
    parser.add_argument(
        "path",
        type=str,
        help="""Path to the parsed file.""",
    )
    parser.add_argument(
        "--db-type",
        default="sqlite",
        type=str,
        choices=("mysql", "sqlite"),
        help="Type of the database.",
    )
    return parser


def run(
    *,
    db_url: str,
    path: str,
    db_type: str,
) -> None:
    """Add an entry to the database.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    import pickle  # nosec

    import sqlalchemy

    from bluesearch.database.identifiers import generate_uid
    from bluesearch.utils import load_spacy_model

    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type == "mysql":
        raise NotImplementedError

    else:
        # This branch never reached because of `choices` in `argparse`
        raise ValueError(f"Unrecognized database type {db_type}.")  # pragma: nocover

    with open(path, "rb") as f:
        article = pickle.load(f)  # nosec

    # Article table.

    # TODO At the moment, no identifiers are extracted. This is a patch waiting for it.
    article_id = generate_uid((article.title,))

    article_mapping = {
        "article_id": article_id,
        "title": article.title,
        "authors": ", ".join(article.authors),
        "abstract": "\n".join(article.abstract),
    }
    article_keys = article_mapping.keys()
    article_fields = ", ".join(article_keys)
    article_binds = f":{', :'.join(article_keys)}"

    with engine.begin() as con:
        query = sqlalchemy.text(
            f"INSERT INTO articles({article_fields}) VALUES({article_binds})"
        )
        con.execute(query, article_mapping)

    # Sentence table.

    sentence_mappings = []
    swapped = ((text, section) for section, text in article.section_paragraphs)
    nlp = load_spacy_model("en_core_sci_lg", disable=["ner"])
    for pposition, (document, section) in enumerate(nlp.pipe(swapped, as_tuples=True)):
        for sposition, sentence in enumerate(document.sents):
            sentence_mapping = {
                "section_name": section,
                "text": sentence.text,
                "article_id": article_id,
                "paragraph_pos_in_article": pposition,
                "sentence_pos_in_paragraph": sposition,
            }
            sentence_mappings.append(sentence_mapping)

    sentences_keys = [
        "section_name",
        "text",
        "article_id",
        "paragraph_pos_in_article",
        "sentence_pos_in_paragraph",
    ]
    sentences_fields = ", ".join(sentences_keys)
    sentences_binds = f":{', :'.join(sentences_keys)}"

    with engine.begin() as con:
        for sentence_mapping in sentence_mappings:
            query = sqlalchemy.text(
                f"INSERT INTO sentences({sentences_fields}) VALUES({sentences_binds})"
            )
            con.execute(query, sentence_mapping)

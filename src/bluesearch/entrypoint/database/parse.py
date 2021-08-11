"""Parsing an article."""
import argparse
import json
import pickle

import sqlalchemy

import bluesearch.database.article as article_module


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Parse article.",
    )
    parser.add_argument(
        "parser",
        type=str,
        help="""Parser class.""",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="""Path to the file/directory to be parsed.""",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="""Path where the parsed article is saved.""",
    )
    return parser


def run(
    *,
    parser: str,
    input_path: str,
    output_path: str,
) -> None:
    """Parse an article.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    valid_parsers = [x.__name__ for x in article_module.ArticleParser.__subclasses__()]

    if parser not in valid_parsers:
        raise ValueError(f"Unsupported parser {parser}. Valid parsers: {valid_parsers}")

    parser_cls = getattr(article_module, parser)

    # We should unify this somehow to make sure all parsers have the same constructor
    if parser == "CORD19ArticleParser":
        with open(input_path) as f:
            parser_inst = parser_cls(json.load(f))
    else:
        parser_inst = parser_cls(input_path)  # not covered since we do not have other parsers

    article = article_module.Article.parse(parser_inst)

    with open(output_path, "wb") as f:
        pickle.dump(article, f)

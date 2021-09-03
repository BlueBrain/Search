"""Parsing an article."""
import argparse
import json
import pickle  # nosec

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    PubmedXMLParser,
)


def get_parser() -> argparse.ArgumentParser:
    """Create a parser."""
    parser = argparse.ArgumentParser(
        description="Parse article.",
    )
    parser.add_argument(
        "parser",
        type=str,
        choices=("CORD19ArticleParser", "PubmedXMLParser"),
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
    parser_inst: ArticleParser
    if parser == "CORD19ArticleParser":
        with open(input_path) as f_input:
            parser_inst = CORD19ArticleParser(json.load(f_input))
    elif parser == "PubmedXMLParser":
        parser_inst = PubmedXMLParser(input_path)

    else:
        raise ValueError(f"Unsupported parser {parser}")

    article = Article.parse(parser_inst)

    with open(output_path, "wb") as f_output:
        pickle.dump(article, f_output)

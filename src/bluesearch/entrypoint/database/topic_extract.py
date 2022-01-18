# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Extract topic of articles."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from bluesearch.database.article import ArticleSource

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the add subcommand.

    Parameters
    ----------
    parser
        The argument parser to initialise.

    Returns
    -------
    argparse.ArgumentParser
        The initialised argument parser. The same object as the `parser`
        argument.
    """
    parser.description = "Extract topic of articles."

    parser.add_argument(
        "source",
        choices=[member.value for member in ArticleSource],
        help="""
        Format of the input.
        If extracting topic of several articles, all articles must have the same format.
        """,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="""
        Path to a file or directory. If a directory, topic will be extracted for
        all articles inside the directory.
        """,
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="""
        Path to the file where the topic information will be written.
        If it does not exist yet, the file is created.
        """,
    )
    parser.add_argument(
        "-m",
        "--match-filename",
        type=str,
        help="""
        Extract topic only of articles with a name matching the given regular
        expression. Ignored when 'input_path' is a path to a file.
        """,
    )
    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="""
        Find articles recursively.
        """,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="""
        If output_file exists and overwrite is true, the output file is overwritten.
        Otherwise, the topic extraction results are going
        to be appended to the `output_file`.
        """,
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="""
        Display files to parse without parsing them.
        Especially useful when using '--match-filename' and / or '--recursive'.
        """,
    )
    return parser


def run(
    *,
    source: str,
    input_path: Path,
    output_file: Path,
    match_filename: str | None,
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Extract topic of articles.

    Parameter description and potential defaults are documented inside of the
    `get_parser` function.
    """
    import datetime

    from defusedxml import ElementTree

    import bluesearch
    from bluesearch.database.topic import (
        extract_article_topics_for_pubmed_article,
        extract_article_topics_from_medrxiv_article,
        extract_journal_topics_for_pubmed_article,
        get_topics_for_arxiv_articles,
        get_topics_for_pmc_article,
    )
    from bluesearch.database.topic_info import TopicInfo
    from bluesearch.utils import JSONL, find_files

    try:
        inputs = find_files(input_path, recursive, match_filename)
    except ValueError:
        logger.error(
            "Argument 'input_path' should be a path "
            "to an existing file or directory!"
        )
        return 1

    if dry_run:
        # Inputs are already sorted.
        print(*inputs, sep="\n")
        return 0

    article_source = ArticleSource(source)
    all_results: list[dict[str, Any]] = []
    if article_source == ArticleSource.PMC:
        for path in inputs:
            logger.info(f"Processing {path}")
            topic_info = TopicInfo(source=article_source, path=path.resolve())
            journal_topics = get_topics_for_pmc_article(path)
            if journal_topics:
                topic_info.add_journal_topics("MeSH", journal_topics)
            all_results.append(topic_info.json())
    elif article_source == ArticleSource.PUBMED:
        for path in inputs:
            logger.info(f"Processing {path}")
            articles = ElementTree.parse(input_path)
            for i, article in enumerate(articles.iter("PubmedArticle")):
                topic_info = TopicInfo(
                    source=article_source,
                    path=path.resolve(),
                    element_in_file=i,
                )
                article_topics = extract_article_topics_for_pubmed_article(article)
                journal_topics = extract_journal_topics_for_pubmed_article(article)
                if article_topics:
                    topic_info.add_article_topics("MeSH", article_topics)
                if journal_topics:
                    topic_info.add_journal_topics("MeSH", journal_topics)
                all_results.append(topic_info.json())
    elif article_source == ArticleSource.ARXIV:
        for path, article_topics in get_topics_for_arxiv_articles(inputs).items():
            topic_info = TopicInfo(source=article_source, path=path)
            topic_info.add_article_topics("arXiv", article_topics)
            all_results.append(topic_info.json())
    elif source in {"biorxiv", "medrxiv"}:
        for path in inputs:
            logger.info(f"Processing {path}")
            topic, journal = extract_article_topics_from_medrxiv_article(path)
            all_results.append(
                {
                    "source": journal,
                    "path": str(path.resolve()),
                    "topics": {
                        "article": {
                            "Subject Area": topic,
                        },
                    },
                    "metadata": {
                        "created-date": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "bbs-version": bluesearch.version.__version__,
                    },
                }
            )

        pass
    else:
        logger.error(f"The source type {source!r} is not implemented yet")
        return 1

    JSONL.dump_jsonl(all_results, output_file, overwrite)

    return 0

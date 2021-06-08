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
"""Abstraction of scientific article data and related tools."""
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Sequence, Tuple


class ArticleParser(ABC):
    """An abstract base class for article parsers."""

    @abstractmethod
    def get_title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """

    @abstractmethod
    def iter_authors(self) -> Generator[str, Any, None]:
        """Iterate over all author names.

        Yields
        ------
        str
            Every author.
        """

    @abstractmethod
    def get_abstract(self) -> Sequence[str]:
        """Get a sequence of paragraphs in the article abstract.

        Returns
        -------
        sequence of str
            The paragraphs of the article abstract.
        """

    @abstractmethod
    def iter_paragraphs(self) -> Generator[Tuple[str, str], Any, None]:
        """Iterate over all paragraphs and titles of sections they are part of.

        Yields
        ------
        str
            The section title.
        str
            The paragraph content.
        """


class CORD19ArticleParser(ArticleParser):
    """Parser for CORD-19 JSON files.

    Parameters
    ----------
    json_file
        The contents of a JSON-file from the CORD-19 database.
    """

    def __init__(self, json_file: dict) -> None:
        # data is a reference to json_file, so we shouldn't modify its contents
        self.data = json_file

        # Check top-level keys
        top_level_keys = {
            "paper_id",
            "metadata",
            "abstract",
            "body_text",
            "bib_entries",
            "ref_entries",
            "back_matter",
        }
        if set(json_file.keys()) != top_level_keys:
            raise ValueError(
                "Incomplete JSON file. Missing keys: "
                f"{top_level_keys - set(json_file.keys())}"
            )

    def get_title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        return self.data["metadata"]["title"]

    def iter_authors(self) -> Generator[str, Any, None]:
        """Iterate over all author names.

        Yields
        ------
        str
            Every author.
        """
        for author in self.data["metadata"]["authors"]:
            author_str = " ".join(
                filter(
                    lambda part: part != "",
                    (
                        author["first"],
                        " ".join(author["middle"]),
                        author["last"],
                        author["suffix"],
                    ),
                )
            )
            yield author_str

    def get_abstract(self) -> List[str]:
        """Get a sequence of paragraphs in the article abstract.

        Returns
        -------
        list of str
            The paragraphs of the article abstract.
        """
        return [paragraph["text"] for paragraph in self.data["abstract"]]

    def iter_paragraphs(self) -> Generator[Tuple[str, str], Any, None]:
        """Iterate over all paragraphs and titles of sections they are part of.

        Yields
        ------
        str
            The section title.
        str
            The paragraph content.
        """
        for paragraph in self.data["body_text"]:
            yield paragraph["section"], paragraph["text"]

    def __str__(self):
        """Get the string representation the the parser instance."""
        return f'CORD-19 article ID={self.data["paper_id"]}'


class Article:
    """Abstraction of a scientific article and its contents."""

    def __init__(self) -> None:
        self.title = ""
        self.authors: List[str] = []
        self.abstract: List[str] = []
        # We require py37+ so this dict is guaranteed insertion-ordered.
        self.sections: Dict[str, List[str]] = {}
        self.parsed = False

    def parse(self, parser: ArticleParser) -> None:
        """Parse an article through a parser.

        Parameters
        ----------
        parser
            An article parser instance.

        Warns
        -----
        UserWarning
            If duplicate section titles are encountered. Since the parsing is
            paragraph-wise duplicate sections are recognized as two non-adjacent
            blocks of paragraphs with the same title.
        """
        if self.parsed:
            raise RuntimeError("Can only parse the article once")
        self.title = parser.get_title()
        self.authors.extend(parser.iter_authors())
        self.abstract.extend(parser.get_abstract())

        # Collect sections
        current_section = None
        for section, paragraph in parser.iter_paragraphs():
            if section not in self.sections:
                self.sections[section] = []
            elif current_section != section:
                # We've already seen that section title and it was not part of
                # the current section.
                warnings.warn(
                    f"Duplicate section titles found in {parser}. The respective"
                    "sections will be merged into one."
                )
            current_section = section
            self.sections[section].append(paragraph)

        self.parsed = True

    def iter_paragraphs(self, with_abstract: bool = True) -> Generator[str, Any, None]:
        """Iterate over all paragraphs in the article.

        Parameters
        ----------
        with_abstract : bool
            If true the abstract paragraphs will be included at the beginning.

        Yields
        ------
        str
            One of the article paragraphs.
        """
        if with_abstract:
            yield from self.abstract
        for section_paragraphs in self.sections.values():
            yield from section_paragraphs

    def __str__(self) -> str:
        """Get a short summary of the article statistics.

        Returns
        -------
        str
            A summary of the article statistics.
        """
        info_str = (
            f'Title    : "{self.title}"\n'
            f'Authors  : {", ".join(self.authors)}\n'
            f"Abstract : {len(self.abstract)} paragraph(s), "
            f"{sum(map(len, self.abstract))} characters\n"
            f"Sections : {len(self.sections)} section(s) "
            f"{sum(sum(map(len, section)) for section in self.sections.values())} "
            f"characters\n"
        )
        for section in self.sections:
            info_str += f"- {section}\n"
        info_str += f"Total text length : {sum(map(len, self.iter_paragraphs()))}\n"

        return info_str.strip()

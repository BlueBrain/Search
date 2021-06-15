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
from dataclasses import dataclass
from typing import Dict, Generator, List, Mapping, Sequence, Tuple, Type, TypeVar

# This is for annotating the return value of the Article.parse class method, see
# https://github.com/python/typing/issues/254#issuecomment-661803922
_T = TypeVar("_T", bound="Article")


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
    def iter_authors(self) -> Generator[str, None, None]:
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
    def iter_paragraphs(self) -> Generator[Tuple[str, str], None, None]:
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
        # the spec also includes "abstract" but it's missing from the PMC parses
        top_level_keys = {
            "paper_id",
            "metadata",
            "body_text",
            "bib_entries",
            "ref_entries",
            "back_matter",
        }
        if not top_level_keys.issubset(json_file.keys()):
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

    def iter_authors(self) -> Generator[str, None, None]:
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
        if "abstract" not in self.data:
            return []

        return [paragraph["text"] for paragraph in self.data["abstract"]]

    def iter_paragraphs(self) -> Generator[Tuple[str, str], None, None]:
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
        # We've always included figure/table captions like this
        for ref_entry in self.data["ref_entries"].values():
            yield "Caption", ref_entry["text"]

    def __str__(self):
        """Get the string representation the the parser instance."""
        return f'CORD-19 article ID={self.data["paper_id"]}'


@dataclass(frozen=True)
class Article:
    """Abstraction of a scientific article and its contents."""

    title: str
    authors: Sequence[str]
    abstract: Sequence[str]
    # We require py37+ so dict is guaranteed insertion-ordered.
    sections: Mapping[str, Sequence[str]]

    @classmethod
    def parse(cls: Type[_T], parser: ArticleParser) -> _T:
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
        title = parser.get_title()
        authors = list(parser.iter_authors())
        abstract = parser.get_abstract()
        sections: Dict[str, List[str]] = {}

        # Collect sections
        current_section = None
        for section, paragraph in parser.iter_paragraphs():
            if section not in sections:
                sections[section] = []
            elif current_section != section:
                # We've already seen that section title and it was not part of
                # the current section.
                warnings.warn(
                    f"Duplicate section titles found in {parser}. The respective"
                    "sections will be merged into one."
                )
            current_section = section
            sections[section].append(paragraph)

        return cls(title, authors, abstract, sections)

    def iter_paragraphs(self, with_abstract: bool = True) -> Generator[str, None, None]:
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

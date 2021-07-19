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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Sequence, Tuple, Type, TypeVar, Union

from lxml import etree  # nosec

# This is for annotating the return value of the Article.parse class method, see
# https://github.com/python/typing/issues/254#issuecomment-661803922
_T = TypeVar("_T", bound="Article")


class ArticleParser(ABC):
    """An abstract base class for article parsers."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """

    @property
    @abstractmethod
    def authors(self) -> Iterable[str]:
        """Get all author names.

        Returns
        -------
        iterable of str
            All authors.
        """

    @property
    @abstractmethod
    def abstract(self) -> Iterable[str]:
        """Get a sequence of paragraphs in the article abstract.

        Returns
        -------
        iterable of str
            The paragraphs of the article abstract.
        """

    @property
    @abstractmethod
    def paragraphs(self) -> Iterable[Tuple[str, str]]:
        """Get all paragraphs and titles of sections they are part of.

        Returns
        -------
        iterable of (str, str)
            For each paragraph a tuple with two strings is returned. The first
            is the section title, the second the paragraph content.
        """


class PubmedXMLParser(ArticleParser):
    """Parser for PubMed XML files using the JATS Journal Publishing DTD.

    Parameters
    ----------
    path
        The path to the XML file from PubMed.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.content = etree.parse(str(path))

    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        titles = self.content.find("//title-group/article-title")
        if titles is None:
            return ""
        return self.text_content(titles)

    @property
    def authors(self) -> Generator[str, None, None]:
        """Get all author names.

        Yields
        ------
        str
            Every author, in the format "Given_Name(s) Surname".
        """
        authors = self.content.xpath('//contrib-group/contrib[@contrib-type="author"]')
        for author in authors:
            try:
                given_names = (
                    self.text_content(author.find("name/given-names")) or ""
                )
                surname = self.text_content(author.find("name/surname")) or ""
                author_str = given_names + " " + surname
                yield author_str.strip()
            except AttributeError:
                # In rare cases, an author may not have a given name or a surname,
                # e.g. it could be an organization. We decide to skip those.
                continue

    @property
    def abstract(self) -> Generator[str, None, None]:
        """Get a sequence of paragraphs in the article abstract.

        Yields
        ------
        paragraph_text
            The paragraphs of the article abstract.
        """
        abstract_pars = self.content.findall("//abstract//p")
        for paragraph in abstract_pars:
            yield self.text_content(paragraph) or ""

    @property
    def paragraphs(self) -> Generator[Tuple[str, str], None, None]:
        """Get all paragraphs and titles of sections they are part of.

        Paragraphs can be parts of text body, or figure or table captions.

        Yields
        ------
        section_title
            The section title.
        paragraph_text
            The paragraph content.
        """
        # Paragraphs of text body
        paragraphs = self.content.xpath("//body//p[not(parent::caption)]")
        for paragraph in paragraphs:
            text = self.text_content(paragraph)
            section = paragraph.find("../title")
            if section is not None:
                section = self.text_content(section)
            yield section or "", text

        # Figure captions
        figs = self.content.findall("//body//fig")
        if figs is not None:
            for fig in figs:
                try:
                    fig_captions = fig.find("caption").getchildren()
                    caption = " ".join([self.text_content(c) for c in fig_captions])
                    yield "Figure Caption", caption
                except AttributeError:
                    continue

        # Table captions
        tables = self.content.xpath("//body//table-wrap")
        if tables is not None:
            for table in tables:
                if table.find("caption/p") is not None:
                    caption = self.text_content(table.find("caption/p"))
                elif table.find("caption/title") is not None:
                    caption = self.text_content(table.find("caption/title"))
                else:
                    caption = ""
                yield "Table Caption", caption

    @staticmethod
    def text_content(element: etree._Element) -> str:
        """Extract all text of an element and of its descendants (at any depth).

        Parameters
        ----------
        element

        Returns
        -------
        text
            Entire text of the element and its descendants.
        """
        return "".join(t for t in element.itertext())


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

    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        return self.data["metadata"]["title"]

    @property
    def authors(self) -> Generator[str, None, None]:
        """Get all author names.

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

    @property
    def abstract(self) -> List[str]:
        """Get a sequence of paragraphs in the article abstract.

        Returns
        -------
        list of str
            The paragraphs of the article abstract.
        """
        if "abstract" not in self.data:
            return []

        return [paragraph["text"] for paragraph in self.data["abstract"]]

    @property
    def paragraphs(self) -> Generator[Tuple[str, str], None, None]:
        """Get all paragraphs and titles of sections they are part of.

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
        """Get the string representation of the parser instance."""
        return f'CORD-19 article ID={self.data["paper_id"]}'


@dataclass(frozen=True)
class Article:
    """Abstraction of a scientific article and its contents."""

    title: str
    authors: Sequence[str]
    abstract: Sequence[str]
    section_paragraphs: Sequence[Tuple[str, str]]

    @classmethod
    def parse(cls: Type[_T], parser: ArticleParser) -> _T:
        """Parse an article through a parser.

        Parameters
        ----------
        parser
            An article parser instance.
        """
        title = parser.title
        authors = tuple(parser.authors)
        abstract = tuple(parser.abstract)
        section_paragraphs = tuple(parser.paragraphs)

        return cls(title, authors, abstract, section_paragraphs)

    def iter_paragraphs(
        self, with_abstract: bool = False
    ) -> Generator[Tuple[str, str], None, None]:
        """Iterate over all paragraphs in the article.

        Parameters
        ----------
        with_abstract : bool
            If true the abstract paragraphs will be included at the beginning.

        Yields
        ------
        str
            Section title of the section the paragraph is in.
        str
            The paragraph text.
        """
        if with_abstract:
            for paragraph in self.abstract:
                yield "Abstract", paragraph
        yield from self.section_paragraphs

    def __str__(self) -> str:
        """Get a short summary of the article statistics.

        Returns
        -------
        str
            A summary of the article statistics.
        """
        # Collection information on text/paragraph lengths
        abstract_length = sum(map(len, self.abstract))
        section_lengths = {}
        for section_title, text in self.section_paragraphs:
            if section_title not in section_lengths:
                section_lengths[section_title] = 0
            section_lengths[section_title] += len(text)
        main_text_length = sum(section_lengths.values())
        all_text_length = abstract_length + main_text_length

        # Construct the return string
        info_str = (
            f'Title    : "{self.title}"\n'
            f'Authors  : {", ".join(self.authors)}\n'
            f"Abstract : {len(self.abstract)} paragraph(s), "
            f"{abstract_length} characters\n"
            f"Sections : {len(section_lengths)} section(s) "
            f"{main_text_length} characters\n"
        )
        for section in section_lengths:
            info_str += f"- {section}\n"
        info_str += f"Total text length : {all_text_length}\n"

        return info_str.strip()

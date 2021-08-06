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
from typing import (
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from xml.etree.ElementTree import Element  # nosec

from defusedxml import ElementTree

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
        super().__init__()
        self.content = ElementTree.parse(str(path))

    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        titles = self.content.find("//title-group/article-title")
        return self.text_content(titles)

    @property
    def authors(self) -> Generator[str, None, None]:
        """Get all author names.

        Yields
        ------
        str
            Every author, in the format "Given_Name(s) Surname".
        """
        authors = self.content.findall(
            "//contrib-group/contrib" '[@contrib-type="author"]'
        )
        for author in authors:
            given_names = self.text_content(author.find("name/given-names"))
            surname = self.text_content(author.find("name/surname"))
            if given_names == "" or surname == "":
                # In rare cases, an author may not have a given name or a surname,
                # e.g. it could be an organization. We decide to skip those.
                continue
            author_str = given_names + " " + surname
            yield author_str.strip()

    @property
    def abstract(self) -> Generator[str, None, None]:
        """Get a sequence of paragraphs in the article abstract.

        Yields
        ------
        str
            The paragraphs of the article abstract.
        """
        abstract_pars = self.content.findall("//abstract//p")
        for paragraph in abstract_pars:
            yield self.text_content(paragraph)

    @property
    def paragraphs(self) -> Generator[Tuple[str, str], None, None]:
        """Get all paragraphs and titles of sections they are part of.

        Paragraphs can be parts of text body, or figure or table captions.

        Yields
        ------
        section : str
            The section title.
        text : str
            The paragraph content.
        """
        # Paragraphs of text body
        abstract_p = self.content.findall(".//abstract/p")
        caption_p = self.content.findall(".//caption/p")
        acknowlegd_p = self.content.findall(".//ack/p")

        exclude_list = abstract_p + caption_p + acknowlegd_p

        section_dirs = self.get_sections()
        for paragraph in self.content.findall(".//p"):
            if paragraph not in exclude_list:
                text = self.text_content(paragraph)
                section_title = ""
                if paragraph in section_dirs:
                    section_title = section_dirs[paragraph]
                yield section_title, text

        # Figure captions
        figs = self.content.findall("//body//fig") or []
        for fig in figs:
            fig_captions = fig.find("caption")
            if fig_captions is None:
                continue
            caption = " ".join(self.text_content(c) for c in fig_captions.getchildren())
            yield "Figure Caption", caption

        # Table captions
        tables = self.content.findall("//body//table-wrap") or []
        for table in tables:
            caption_element = table.find("caption/p") or table.find("caption/title")
            caption = self.text_content(caption_element)
            yield "Table Caption", caption

    def get_sections(self) -> dict:
        """Extract sections information.

        Returns
        -------
        sections_dir : dict
            Dictionary whose keys are paragraphs and value are section title.
        """
        sections_dir = {}
        for sec in self.content.findall(".//sec"):

            section_title = ""
            section_paragraphs = []

            for element in sec.getchildren():

                if element.tag == "title":
                    section_title = self.text_content(element)
                elif element.tag == "p":
                    section_paragraphs.append(element)
                else:
                    continue

            for paragraph in section_paragraphs:
                sections_dir[paragraph] = section_title

        return sections_dir

    @staticmethod
    def text_content(element: Optional[Element]) -> str:
        """Extract all text of an element and of its descendants (at any depth).

        Parameters
        ----------
        element : etree._Element or None
            XML element to parse.

        Returns
        -------
        str
            Entire text of the element and its descendants.
        """
        if element is None:
            return ""
        else:
            return "".join(element.itertext())


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

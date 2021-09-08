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
from __future__ import annotations

import html
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Sequence
from xml.etree.ElementTree import Element  # nosec

from defusedxml import ElementTree


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
    def paragraphs(self) -> Iterable[tuple[str, str]]:
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

    def __init__(self, path: str | Path) -> None:
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
        titles = self.content.find(".//title-group/article-title")
        return self._element_to_str(titles)

    @property
    def authors(self) -> Generator[str, None, None]:
        """Get all author names.

        Yields
        ------
        str
            Every author, in the format "Given_Name(s) Surname".
        """
        authors = self.content.findall(
            ".//contrib-group/contrib[@contrib-type='author']"
        )
        for author in authors:
            given_names = self._element_to_str(author.find("name/given-names"))
            surname = self._element_to_str(author.find("name/surname"))
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
        abstract_pars = self.content.findall(".//abstract//p")
        for paragraph in abstract_pars:
            yield self._element_to_str(paragraph)

    @property
    def paragraphs(self) -> Generator[tuple[str, str], None, None]:
        """Get all paragraphs and titles of sections they are part of.

        Paragraphs can be parts of text body, or figure or table captions.

        Yields
        ------
        section : str
            The section title.
        text : str
            The paragraph content.
        """
        # Abstract are removed because already parsed in abstract property.
        abstract_p = self.content.findall(".//abstract/p")
        # Caption are removed because already parsed later in the current method
        # with the section name "Caption".
        caption_p = self.content.findall(".//caption/p")
        # Acknowledgements are removed because not useful.
        acknowledg_p = self.content.findall(".//ack/p")

        exclude_list = abstract_p + caption_p + acknowledg_p

        # Paragraphs of text body
        section_dirs = self.get_paragraphs_sections_mapping()
        for paragraph in self.content.findall(".//p"):
            if paragraph not in exclude_list:
                text = self._element_to_str(paragraph)
                section_title = ""
                if paragraph in section_dirs:
                    section_title = section_dirs[paragraph]
                yield section_title, text

        # Figure captions
        figs = self.content.findall(".//fig")
        for fig in figs:
            fig_captions = fig.findall("caption")
            if fig_captions is None:
                continue
            caption = " ".join(self._element_to_str(c) for c in list(fig_captions))
            yield "Figure Caption", caption

        # Table captions
        tables = self.content.findall(".//table-wrap")
        for table in tables:
            caption_elements = table.findall(".//caption/p") or table.findall(
                ".//caption/title"
            )
            if caption_elements is None:
                continue
            caption = " ".join(self._element_to_str(c) for c in caption_elements)
            yield "Table Caption", caption

    def get_paragraphs_sections_mapping(self) -> dict[Element, str]:
        """Construct mapping between all paragraphs and their section name.

        Returns
        -------
        mapping : dict
            Dictionary whose keys are paragraphs and value are section title.
        """
        mapping = {}
        for sec in self.content.findall(".//sec"):

            section_title = ""
            section_paragraphs = []

            for element in sec:

                if element.tag == "title":
                    section_title = self._element_to_str(element)
                elif element.tag == "p":
                    section_paragraphs.append(element)

            for paragraph in section_paragraphs:
                mapping[paragraph] = section_title

        return mapping

    def _inner_text(self, element: Element) -> str:
        """Convert all inner text and sub-elements to one string.

        In short, we collect all the inner text while also converting all
        sub-elements that we encounter to strings using ``self._element_to_str``.
        All escaped HTML in the raw text is unescaped.

        For example, if schematically the element is given by

            element = "<p>I <bold>like</bold> python &amp; ice cream.<p>"

        then ``_inner_text(element)`` would give

            "I like python & ice cream."

        provided that "<bold>like</bold>" is resolved to "like" by the
        ``self._element_to_str`` method.

        Parameters
        ----------
        element
            The input XML element.

        Returns
        -------
        str
            The inner text and sub-elements converted to one single string.
        """
        parts = []
        text = html.unescape(element.text or "")
        if text:
            parts.append(text)
        for sub_element in element:
            # recursively parse the sub-element
            parts.append(self._element_to_str(sub_element))
            # don't forget the text after the sub-element
            text = html.unescape(sub_element.tail or "")
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def _element_to_str(self, element: Element | None) -> str:
        """Convert an element and all its contents to a string.

        Parameters
        ----------
        element
            The input XML element.

        Returns
        -------
        str
            A parsed string representation of the input XML element.
        """
        if element is None:
            return ""

        if element.tag in {"p", "bold", "italic", "underline", "monospace", "xref"}:
            # Currently this is the same as the default handling. Writing it out
            # explicitly here to decouple from the default handling, which may
            # change in the future.
            return self._inner_text(element)
        elif element.tag == "sub":
            return f"_{self._inner_text(element)}"
        elif element.tag == "sup":
            return f"^{self._inner_text(element)}"
        elif element.tag == "inline-formula":
            return "FORMULA"
        elif element.tag == "disp-formula":
            return "\nFORMULA-BLOCK"
        elif element.tag in {"ext-link", "uri"}:
            return "URL"
        elif element.tag == "email":
            return "EMAIL"
        else:
            # Default handling for all other element tags
            return self._inner_text(element)


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
    def abstract(self) -> list[str]:
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
    def paragraphs(self) -> Generator[tuple[str, str], None, None]:
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
    section_paragraphs: Sequence[tuple[str, str]]

    @classmethod
    def parse(cls, parser: ArticleParser) -> Article:
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
    ) -> Generator[tuple[str, str], None, None]:
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

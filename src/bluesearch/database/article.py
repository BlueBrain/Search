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
import re
import string
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Sequence, Tuple
from xml.etree.ElementTree import Element  # nosec

from defusedxml import ElementTree
from mashumaro import DataClassJSONMixin

from bluesearch.database.identifiers import generate_uid


def get_arxiv_id(path: str | Path) -> str | None:
    """Compute arXiv ID, including version, from file path.

    Parameters
    ----------
    path
        The file path to an arXiv article.

    Returns
    -------
    str or None
        arXiv ID, if possible to compute.

    Raises
    ------
    ValueError
        If no valid arXiv ID could be inferred from the file path.

    References
    ----------
    https://arxiv.org/help/arxiv_identifier
    """
    path = Path(path)

    # New format, since 2007-04, only needs path stem:
    # - since 2015-01 have format YYMM.NNNNN (i.e. 5 digits)
    # - up to 2014-12 have format YYMM.NNNN (i.e. 4 digits)
    pattern = re.compile(r"\A(\d{4}\.\d{4}\d?v\d+)\Z")
    match = re.search(pattern, path.stem)
    if match:
        return f"arxiv:{match.groups()[0]}"

    # Old format, up to 2007-03, needs to look at the whole path:
    # - some_path/arxiv/<archive>/<format>/YYMM/YYMMNNNv<version>.<ext>
    # Note: format may contain "-"
    pattern = re.compile(r"arxiv/([\w-]+)/\w+/\d{4}/(\d{7}v\d+)\.\w+\Z")
    match = re.search(pattern, "/".join(path.parts[-5:]))
    if match:
        match_groups = match.groups()
        return f"arxiv:{match_groups[0]}/{match_groups[1]}"

    raise ValueError(f"Could not extract arXiv ID from file path {path}")


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

    @property
    def pubmed_id(self) -> Optional[str]:
        """Get Pubmed ID.

        Returns
        -------
        str or None
            Pubmed ID if specified, otherwise None.
        """
        return None

    @property
    def pmc_id(self) -> Optional[str]:
        """Get PMC ID.

        Returns
        -------
        str or None
            PMC ID if specified, otherwise None.
        """
        return None

    @property
    def arxiv_id(self) -> Optional[str]:
        """Get arXiv ID.

        Returns
        -------
        str or None
            arXiv ID if specified, otherwise None.
        """
        return None

    @property
    def doi(self) -> Optional[str]:
        """Get DOI.

        Returns
        -------
        str or None
            DOI if specified, otherwise None.
        """
        return None

    @property
    def uid(self) -> Optional[str]:
        """Generate unique ID of the article based on different identifiers.

        Returns
        -------
        str or None
            If at least one identifier exists, unique id is created. Otherwise,
            the returned uid is None.
        """
        return generate_uid((self.pubmed_id, self.pmc_id, self.arxiv_id, self.doi))


class JATSXMLParser(ArticleParser):
    """Parser for JATS XML files.

    This could be used for articles from PubMed Central, bioRxiv, and medRxiv.

    Parameters
    ----------
    path
        The path to a JATS XML file.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.content = ElementTree.parse(str(path))
        self.ids = self.get_ids()

    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        titles = self.content.find("./front/article-meta/title-group/article-title")
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
            "./front/article-meta/contrib-group/contrib[@contrib-type='author']"
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
        abstract = self.content.find("./front/article-meta/abstract")
        if abstract:
            for _, text in self.parse_section(abstract):
                yield text

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
        # Paragraphs of text body
        body = self.content.find("./body")
        if body:
            yield from self.parse_section(body)

        # Figure captions
        figs = self.content.findall("./body//fig")
        for fig in figs:
            fig_captions = fig.findall("caption")
            if fig_captions is None:
                continue
            caption = " ".join(self._element_to_str(c) for c in list(fig_captions))
            if caption:
                yield "Figure Caption", caption

        # Table captions
        tables = self.content.findall("./body//table-wrap")
        for table in tables:
            caption_elements = table.findall("./caption/p") or table.findall(
                "./caption/title"
            )
            if caption_elements is None:
                continue
            caption = " ".join(self._element_to_str(c) for c in caption_elements)
            if caption:
                yield "Table Caption", caption

    @property
    def pubmed_id(self) -> Optional[str]:
        """Get Pubmed ID.

        Returns
        -------
        str or None
            Pubmed ID if specified, otherwise None.
        """
        return self.ids.get("pmid")

    @property
    def pmc_id(self) -> Optional[str]:
        """Get PMC ID.

        Returns
        -------
        str or None
            PMC ID if specified, otherwise None.
        """
        return self.ids.get("pmc")

    @property
    def doi(self) -> Optional[str]:
        """Get DOI.

        Returns
        -------
        str or None
            DOI if specified, otherwise None.
        """
        return self.ids.get("doi")

    def get_ids(self) -> dict[str, str]:
        """Get all specified IDs of the paper.

        Returns
        -------
        ids : dict
            Dictionary whose keys are ids type and value are ids values.
        """
        ids = {}
        article_ids = self.content.findall("./front/article-meta/article-id")

        for article_id in article_ids:

            if "pub-id-type" not in article_id.attrib.keys():
                continue

            ids[article_id.attrib["pub-id-type"]] = article_id.text

        return ids

    def parse_section(self, section: Element) -> Generator[tuple[str, str], None, None]:
        """Parse section children depending on the tag.

        Parameters
        ----------
        section
            The input XML element.

        Returns
        -------
        str
            The section title.
        str
            A parsed string representation of the input XML element.
        """
        sec_title = self._element_to_str(section.find("title"))
        for element in section:
            if element.tag == "sec":
                yield from self.parse_section(element)
            elif element.tag in {"title", "caption", "fig", "table-wrap"}:
                continue
            else:
                text = self._element_to_str(element)
                if text:
                    yield sec_title, text

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
        text_parts = [html.unescape(element.text or "")]
        for sub_element in element:
            # recursively parse the sub-element
            text_parts.append(self._element_to_str(sub_element))
            # don't forget the text after the sub-element
            text_parts.append(html.unescape(sub_element.tail or ""))
        return unicodedata.normalize("NFKC", "".join(text_parts)).strip()

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

        if element.tag in {
            "bold",
            "italic",
            "monospace",
            "p",
            "sc",
            "styled-content",
            "underline",
            "xref",
        }:
            # Mostly styling tags for which getting the inner text is enough.
            # Currently this is the same as the default handling. Writing it out
            # explicitly here to decouple from the default handling, which may
            # change in the future.
            return self._inner_text(element)
        elif element.tag == "sub":
            return f"_{self._inner_text(element)}"
        elif element.tag == "sup":
            return f"^{self._inner_text(element)}"
        elif element.tag in {
            "disp-formula",
            "email",
            "ext-link",
            "inline-formula",
            "uri",
        }:
            return ""
        else:
            # Default handling for all other element tags
            return self._inner_text(element)


class PubMedXMLParser(ArticleParser):
    """Parser for PubMed abstract."""

    def __init__(self, data: Element | Path | str) -> None:
        super().__init__()
        self.content: ElementTree
        if isinstance(data, Element):
            self.content = data
        else:
            self.content = ElementTree.parse(str(data))

    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        title = self.content.find("./MedlineCitation/Article/ArticleTitle")
        return title.text

    @property
    def authors(self) -> Iterable[str]:
        """Get all author names.

        Returns
        -------
        iterable of str
            All authors.
        """
        authors = self.content.find("./MedlineCitation/Article/AuthorList")

        if authors is None:
            # No author to parse: stop and return an empty iterable.
            return ()

        for author in authors:
            # Author entries with 'ValidYN' == 'N' are incorrect entries:
            # https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/190101/att-ValidYN.html.
            if author.get("ValidYN") == "Y":
                # 'LastName' is a required field if there is no 'CollectiveName'.
                lastname = author.find("LastName")
                # 'ForeName' is an optional field only used with 'LastName'.
                forenames = author.find("ForeName")

                parts = (forenames, lastname)
                name = [x.text for x in parts if x is not None]
                if len(name) > 0:
                    yield " ".join(name)

    @property
    def abstract(self) -> Iterable[str]:
        """Get a sequence of paragraphs in the article abstract.

        Returns
        -------
        iterable of str
            The paragraphs of the article abstract.
        """
        paragraphs = self.content.find("./MedlineCitation/Article/Abstract")

        if paragraphs is None:
            # No paragraphs to parse: stop and return an empty iterable.
            return ()

        for paragraph in paragraphs.iter("AbstractText"):
            yield paragraph.text

    @property
    def paragraphs(self) -> Iterable[tuple[str, str]]:
        """Get all paragraphs and titles of sections they are part of.

        Returns
        -------
        iterable of (str, str)
            For each paragraph a tuple with two strings is returned. The first
            is the section title, the second the paragraph content.
        """
        # No paragraph to parse in PubMed article sets: return an empty iterable.
        return ()

    @property
    def pubmed_id(self) -> Optional[str]:
        """Get Pubmed ID.

        Returns
        -------
        str or None
            Pubmed ID if specified, otherwise None.
        """
        pubmed_id = self.content.find("./MedlineCitation/PMID")
        return pubmed_id.text

    @property
    def pmc_id(self) -> Optional[str]:
        """Get PMC ID.

        Returns
        -------
        str or None
            PMC ID if specified, otherwise None.
        """
        pmc_id = self.content.find(
            "./PubmedData/ArticleIdList/ArticleId[@IdType='pmc']"
        )
        return None if pmc_id is None else pmc_id.text

    @property
    def doi(self) -> Optional[str]:
        """Get DOI.

        Returns
        -------
        str or None
            DOI if specified, otherwise None.
        """
        doi = self.content.find("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
        return None if doi is None else doi.text


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

    @property
    def pmc_id(self) -> Optional[str]:
        """Get PMC ID.

        Returns
        -------
        str or None
            PMC ID if specified, otherwise None.
        """
        return self.data.get("paper_id")

    def __str__(self):
        """Get the string representation of the parser instance."""
        return f'CORD-19 article ID={self.data["paper_id"]}'


class TEIXMLParser(ArticleParser):
    """Parser for TEI XML files.

    Parameters
    ----------
    path
        The path to a TEI XML file.
    is_arxiv
        Set to `True` if the TEI XML file was generated by parsing an arXiv PDF.
    """

    def __init__(self, path: str | Path, is_arxiv: Optional[bool] = False):
        path = Path(path)
        with path.open() as fp:
            self.content = ElementTree.fromstring(fp.read())
        self.tei_namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
        self._tei_ids: dict[str, str] | None = None
        self._arxiv_id = get_arxiv_id(path) if is_arxiv else None


    @property
    def title(self) -> str:
        """Get the article title.

        Returns
        -------
        str
            The article title.
        """
        title = self.content.find(
            "./tei:teiHeader/tei:fileDesc/tei:titleStmt/", self.tei_namespace
        )
        return self._element_to_str(title)

    @property
    def authors(self) -> Generator[str, None, None]:
        """Get all author names.

        Yields
        ------
        str
            Every author, in the format "Given_Name(s) Surname".
        """
        for pers_name in self.content.findall(
            "./tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic"
            "/tei:author/tei:persName",
            self.tei_namespace,
        ):
            parts = [
                pers_name.find("./tei:forename[@type='first']", self.tei_namespace),
                pers_name.find("./tei:forename[@type='middle']", self.tei_namespace),
                pers_name.find("./tei:surname", self.tei_namespace),
            ]

            parts = [self._element_to_str(part) for part in parts]
            yield " ".join([part for part in parts if part]).strip()

    @property
    def abstract(self) -> Generator[str, None, None]:
        """Get a sequence of paragraphs in the article abstract.

        Yields
        ------
        str
            The paragraphs of the article abstract.
        """
        for div in self.content.findall(
            "./tei:teiHeader/tei:profileDesc/tei:abstract/tei:div",
            self.tei_namespace,
        ):
            yield from self._build_texts(div)

    @property
    def paragraphs(self) -> Generator[tuple[str, str], None, None]:
        """Get all paragraphs and titles of sections they are part of.

        Paragraphs can be parts of text body, or figure or table captions.

        Yields
        ------
        section_title : str
            The section title.
        text : str
            The paragraph content.
        """
        for div in self.content.findall(
            "./tei:text/tei:body/tei:div",
            self.tei_namespace,
        ):
            head = div.find("./tei:head", self.tei_namespace)
            section_title = self._element_to_str(head)
            text_elements = []
            for child in div:
                if not child.tag.endswith("head"):
                    text_elements.append(child)
            for text in self._build_texts(text_elements):
                yield section_title, text

        # Figure and Table Caption
        for figure in self.content.findall(
            "./tei:text/tei:body/tei:figure", self.tei_namespace
        ):
            caption = figure.find("./tei:figDesc", self.tei_namespace)
            caption_str = self._element_to_str(caption)
            if not caption_str:
                continue
            if figure.get("type") == "table":
                yield "Table Caption", caption_str
            else:
                yield "Figure Caption", caption_str

    @property
    def arxiv_id(self) -> Optional[str]:
        """Get arXiv ID.

        Returns
        -------
        str or None
            arXiv ID if specified, otherwise None.
        """
        return self._arxiv_id

    @property
    def doi(self) -> Optional[str]:
        """Get DOI.

        Returns
        -------
        str or None
            DOI if specified, otherwise None.
        """
        return self.tei_ids.get("DOI")

    @property
    def tei_ids(self) -> dict:
        """Extract all IDs of the TEI XML.

        Returns
        -------
        dict
            Dictionary containing all the IDs of the TEI XML content
            with the key being the ID type and the value being the ID value.
        """
        if self._tei_ids is None:
            self._tei_ids = {}
            for idno in self.content.findall(
                "./tei:teiHeader/tei:fileDesc/tei:sourceDesc"
                "/tei:biblStruct/tei:idno",
                self.tei_namespace,
            ):
                id_type = idno.get("type")
                self._tei_ids[id_type] = idno.text

        return self._tei_ids

    @staticmethod
    def _element_to_str(element: Element | None) -> str:
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
        return "".join(element.itertext())

    def _build_texts(self, elements: Iterable[Element]) -> Generator[str, None, None]:
        """Compose paragraphs and formulas to meaningful texts.

        In the abstract and main text of TEI XML parsers one finds a mix of
        <p> and <formula> tags. Several of these tags could be part of one
        sentence. This method tries to reconstruct sentences that are
        partitioned in this way. The formulas are replaced by the FORMULA
        placeholder.

        Parameters
        ----------
        elements
            An iterable of <p> and <formula> elements.

        Yields
        ------
        str
            One or more sentences as one string.

        Raises
        ------
        RuntimeError
            If a tag is encountered that is neither <p> nor <formula>.
        """
        # In TEI XML all tags are prefixed with the namespace.
        ns = self.tei_namespace["tei"]
        prefix = f"{{{ns}}}" if ns else ""
        # At every change ensure that there's no space at the end of text
        text = ""

        def if_non_empty(text_: str) -> Generator[str, None, None]:
            """Yield if text is non-empty and make sure it ends with a period."""
            if text_:
                if not text_.endswith("."):
                    text_ += "."
                yield text_

        for child in elements:
            if child.tag == prefix + "p":
                p_text = self._element_to_str(child).strip()
                if not p_text:
                    continue
                if p_text[0] in string.ascii_uppercase:
                    # The sentence in the text has finished.
                    # Yield and start a new one
                    yield from if_non_empty(text)
                    text = p_text
                else:
                    # The sentence in the text continues
                    text += " " + p_text
            elif child.tag == prefix + "formula":
                # Maybe use FORMULA-BLOCK instead?
                text += " FORMULA"
            else:
                all_text = "".join(self._element_to_str(e) for e in elements)
                raise RuntimeError(
                    f"Unexpected tag: {child.tag}\nall text:\n{all_text}"
                )

        # Yield the last remaining text
        yield from if_non_empty(text)


@dataclass(frozen=True)
class Article(DataClassJSONMixin):
    """Abstraction of a scientific article and its contents."""

    title: str
    authors: Sequence[str]
    abstract: Sequence[str]
    section_paragraphs: Sequence[Tuple[str, str]]
    pubmed_id: Optional[str] = None
    pmc_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    uid: Optional[str] = None

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
        pubmed_id = parser.pubmed_id
        pmc_id = parser.pmc_id
        arxiv_id = parser.arxiv_id
        doi = parser.doi
        uid = parser.uid

        return cls(
            title,
            authors,
            abstract,
            section_paragraphs,
            pubmed_id,
            pmc_id,
            arxiv_id,
            doi,
            uid,
        )

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

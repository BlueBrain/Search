import inspect
import pathlib
import xml.etree.ElementTree
from itertools import chain

import pytest
from defusedxml import ElementTree

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    PubmedXMLParser,
)


class SimpleTestParser(ArticleParser):
    def __init__(self):
        self._title = "Test Title"
        self._authors = ("Author 1", "Author 2")
        self._abstract = ("Abstract paragraph 1", "Abstract paragraph 2")
        self._paragraphs = [
            ("Section 1", "Paragraph 1."),
            ("Section 1", "Paragraph 2."),
            ("Section 2", "Paragraph 1."),
        ]
        self._pubmed_id = "pubmed_id"
        self._pmc_id = "pmc_id"
        self._doi = "doi"
        self._uid = "fake_uid"

    @property
    def title(self):
        return self._title

    @property
    def authors(self):
        yield from self._authors

    @property
    def abstract(self):
        return self._abstract

    @property
    def paragraphs(self):
        yield from self._paragraphs

    @property
    def pubmed_id(self):
        return self._pubmed_id

    @property
    def pmc_id(self):
        return self._pmc_id

    @property
    def doi(self):
        return self._doi

    @property
    def uid(self):
        return self._uid


@pytest.fixture(scope="session")
def pubmed_xml_parser(test_data_path):
    path = pathlib.Path(test_data_path) / "sample_file.xml"
    parser = PubmedXMLParser(path.resolve())
    return parser


class TestPubmedXMLArticleParser:
    def test_init(self, pubmed_xml_parser):
        assert isinstance(pubmed_xml_parser.content, xml.etree.ElementTree.ElementTree)

    def test_title(self, pubmed_xml_parser):
        title = pubmed_xml_parser.title
        assert title == "Article Title"

    def test_authors(self, pubmed_xml_parser):
        authors = pubmed_xml_parser.authors
        assert inspect.isgenerator(authors)
        authors = tuple(authors)

        assert len(authors) == 2
        assert authors[0] == "Author Given Names 1 Author Surname 1"
        assert authors[1] == "Author Given Names 2 Author Surname 2"

    def test_abstract(self, pubmed_xml_parser):
        abstract = pubmed_xml_parser.abstract
        assert inspect.isgenerator(abstract)
        abstract = tuple(abstract)
        assert len(abstract) == 2
        assert abstract[0] == "Abstract Paragraph 1"
        assert abstract[1] == "Abstract Paragraph 2"

    def test_paragraphs(self, pubmed_xml_parser):
        paragraphs = pubmed_xml_parser.paragraphs
        assert inspect.isgenerator(paragraphs)
        paragraphs = tuple(paragraphs)
        assert len(paragraphs) == 7 + 1 + 3  # for paragraph, caption, table

        for i, paragraph in enumerate(paragraphs):
            assert isinstance(paragraph, tuple)
            assert isinstance(paragraph[0], str)
            assert isinstance(paragraph[1], str)
            if i == 7:
                assert paragraph[0] == "Figure Caption"
            if i > 7:
                assert paragraph[0] == "Table Caption"

        assert paragraphs[0] == ("", "Paragraph 1")
        assert paragraphs[3] == ("Section Title 1", "Paragraph Section 1")
        assert paragraphs[4] == ("Section Title 2", "Paragraph Section 2")

    def test_pubmed_id(self, pubmed_xml_parser):
        pubmed_id = pubmed_xml_parser.pubmed_id
        assert isinstance(pubmed_id, str)
        assert pubmed_id == "PMID"

    def test_pmc_id(self, pubmed_xml_parser):
        pmc_id = pubmed_xml_parser.pmc_id
        assert isinstance(pmc_id, str)
        assert pmc_id == "PMC"

    def test_doi(self, pubmed_xml_parser):
        doi = pubmed_xml_parser.doi
        assert isinstance(doi, str)
        assert doi == "DOI"

    def test_uid(self, pubmed_xml_parser):
        uid = pubmed_xml_parser.uid
        assert isinstance(uid, str)
        assert len(uid) == 32

    @pytest.mark.parametrize(
        ("input_xml", "expected_inner_text"),
        (
            ("<p>Simple paragraph.</p>", "Simple paragraph."),
            ("<p>Nested <p>paragraph</p>.</p>", "Nested paragraph."),
            (
                "<p>Paragraph <italic>with</italic> some <bold>styles</bold>.</p>",
                "Paragraph with some styles.",
            ),
            ("<p>Paragraph with &quot;escapes&#34;.</p>", 'Paragraph with "escapes".'),
            (
                "<p><p>Sub-tags</p> at beginning and <p>end</p>.</p>",
                "Sub-tags at beginning and end.",
            ),
            ("<p>My email is <email>me@epfl.ch</email></p>", "My email is EMAIL"),
        ),
    )
    def test_inner_text_extraction(
        self, pubmed_xml_parser, input_xml, expected_inner_text
    ):
        element = ElementTree.fromstring(input_xml)
        inner_text = pubmed_xml_parser._inner_text(element)
        assert inner_text == expected_inner_text

    @pytest.mark.parametrize(
        ("input_xml", "expected_str"),
        (
            ("<p>Simple paragraph.</p>", "Simple paragraph."),
            ("<bold>Bold text</bold>", "Bold text"),
            ("<italic>Italic text</italic>", "Italic text"),
            ("<underline>Underlined text</underline>", "Underlined text"),
            ("<monospace>Monospaced text</monospace>", "Monospaced text"),
            ("<xref>Hawking20</xref>", "Hawking20"),
            ("<sc>Text in small caps</sc>", "Text in small caps"),
            ("<styled-content>Cool style</styled-content>", "Cool style"),
            ("<sub>subbed</sub>", "_subbed"),
            ("<sup>supped</sup>", "^supped"),
            ("<inline-formula>Completely ignored</inline-formula>", "FORMULA"),
            ("<disp-formula>Block formula</disp-formula>", "\nFORMULA-BLOCK"),
            ("<ext-link>https://www.google.com</ext-link>", "URL"),
            ("<uri>file:///path/to/file</uri>", "URL"),
            ("<email>me@domain.ai</email>", "EMAIL"),
            (
                "<unknown-tag>Default: extract inner text.</unknown-tag>",
                "Default: extract inner text.",
            ),
        ),
    )
    def test_element_to_str_works(self, pubmed_xml_parser, input_xml, expected_str):
        element = ElementTree.fromstring(input_xml)
        element_str = pubmed_xml_parser._element_to_str(element)
        assert element_str == expected_str

    def test_element_to_str_of_none(self, pubmed_xml_parser):
        assert pubmed_xml_parser._element_to_str(None) == ""


class TestCORD19ArticleParser:
    def test_init(self, real_json_file):
        # Should be able to read real JSON files no problem.
        parser = CORD19ArticleParser(real_json_file)
        assert parser.data == real_json_file

        # If any of the mandatory top-level keys are missing in the JSON file
        # then an exception should be raised.
        with pytest.raises(ValueError, match="Incomplete JSON file"):
            CORD19ArticleParser({})

    def test_title(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        title = parser.title
        assert title != ""
        assert title == real_json_file["metadata"]["title"]

    def test_authors(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        authors = tuple(parser.authors)

        # Check that all authors have been parsed
        assert len(authors) == len(real_json_file["metadata"]["authors"])

        # Check that all name parts of all authors have been collected
        for author, author_dict in zip(authors, real_json_file["metadata"]["authors"]):
            assert author_dict["first"] in author
            assert author_dict["last"] in author
            assert author_dict["suffix"] in author
            for middle in author_dict["middle"]:
                assert middle in author

    def test_abstract(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        abstract = parser.abstract

        if "abstract" in real_json_file:
            # Check that all paragraphs were parsed
            assert len(abstract) == len(real_json_file["abstract"])

            # Check that all paragraph texts match
            for paragraph, paragraph_dict in zip(abstract, real_json_file["abstract"]):
                assert paragraph == paragraph_dict["text"]
        else:
            # Check that if "abstract" is missing then an empty list is returned.
            # This should be true for all PMC parses.
            assert len(abstract) == 0

    def test_paragraphs(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        paragraphs = tuple(parser.paragraphs)

        # Check that all paragraphs were parsed
        n_body_text = len(real_json_file["body_text"])
        n_ref_entries = len(real_json_file["ref_entries"])
        assert len(paragraphs) == n_body_text + n_ref_entries

        # Check that all paragraph texts match
        for (section, text), paragraph_dict in zip(
            paragraphs, real_json_file["body_text"]
        ):
            assert section == paragraph_dict["section"]
            assert text == paragraph_dict["text"]

    def test_pubmed_id(self, real_json_file):
        # There is no Pubmed ID specified in the schema of CORD19 json files
        parser = CORD19ArticleParser(real_json_file)
        pubmed_id = parser.pubmed_id
        assert pubmed_id is None

    def test_pmc_id(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        pmc_id = parser.pmc_id
        assert isinstance(pmc_id, str)

    def test_doi(self, real_json_file):
        # There is no DOI specified in the schema of CORD19 json files
        parser = CORD19ArticleParser(real_json_file)
        doi = parser.doi
        assert doi is None

    def test_uid(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        uid = parser.uid
        assert isinstance(uid, str)
        assert len(uid) == 32

    def test_str(self, real_json_file):
        parser = CORD19ArticleParser(real_json_file)
        parser_str = str(parser)

        # Should be "CORD-19 article ID=<value>" or similar
        assert "CORD-19" in parser_str
        assert str(real_json_file["paper_id"]) in parser_str


class TestArticle:
    def test_parse(self):
        # Test article parsing
        parser = SimpleTestParser()
        article = Article.parse(parser)
        assert article.title == parser.title
        assert article.authors == tuple(parser.authors)

        # Test iterating over all paragraphs in the article. By default the
        # abstract is not included
        for text, text_want in zip(article.iter_paragraphs(), parser.paragraphs):
            assert text == text_want

        # This time test with the abstract paragraphs included.
        abstract_paragraphs = [("Abstract", text) for text in parser.abstract]
        for text, text_want in zip(
            article.iter_paragraphs(with_abstract=True),
            chain(abstract_paragraphs, parser.paragraphs),
        ):
            assert text == text_want

    def test_str(self):
        parser = SimpleTestParser()
        article = Article.parse(parser)
        article_str = str(article)
        assert parser.title in article_str
        for author in parser.authors:
            assert author in article_str

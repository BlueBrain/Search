import inspect
import pathlib
import xml.etree.ElementTree
from itertools import chain, zip_longest

import pytest
from defusedxml import ElementTree

from bluesearch.database.article import (
    Article,
    ArticleParser,
    CORD19ArticleParser,
    JATSXMLParser,
    PubMedXMLParser,
    TEIXMLParser,
    get_arxiv_id,
)


@pytest.mark.parametrize(
    ("path", "expected_id"),
    (
        ("downloads/arxiv/arxiv/pdf/1802/1802.102998v99.xml", None),
        ("downloads/arxiv/q-bio/pdf/0309/0309.033v2.pdf", None),
        ("downloads/arxiv/arxiv/pdf/1802/1802.10298v99.xml", "arxiv:1802.10298v99"),
        ("downloads/arxiv/arxiv/pdf/1411/1411.7903v4.json", "arxiv:1411.7903v4"),
        ("downloads/arxiv/q-bio/pdf/0309/0309033v2.pdf", "arxiv:q-bio/0309033v2"),
        ("1411.7903v4.xml", "arxiv:1411.7903v4"),
        ("0309033v2.pdf", None),
    ),
)
def test_get_arxiv_id(path, expected_id):
    if expected_id is not None:
        assert get_arxiv_id(path) == expected_id
    else:
        with pytest.raises(ValueError):
            get_arxiv_id(path)


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
def jats_xml_parser(test_data_path):
    path = pathlib.Path(test_data_path) / "jats_article.xml"
    parser = JATSXMLParser(path.resolve())
    return parser


@pytest.fixture(scope="session")
def tei_xml_parser(test_data_path):
    path = pathlib.Path(test_data_path) / "1411.7903v4.xml"
    parser = TEIXMLParser(path)
    return parser


@pytest.fixture(scope="session")
def tei_xml_arxiv_parser(test_data_path):
    path = pathlib.Path(test_data_path) / "1411.7903v4.xml"
    parser = TEIXMLParser(path, is_arxiv=True)
    return parser


class TestJATSXMLArticleParser:
    def test_init(self, jats_xml_parser):
        assert isinstance(jats_xml_parser.content, xml.etree.ElementTree.ElementTree)

    def test_title(self, jats_xml_parser):
        title = jats_xml_parser.title
        assert title == "Article Title"

    def test_authors(self, jats_xml_parser):
        authors = jats_xml_parser.authors
        assert inspect.isgenerator(authors)
        authors = tuple(authors)

        assert len(authors) == 2
        assert authors[0] == "Author Given Names 1 Author Surname 1"
        assert authors[1] == "Author Given Names 2 Author Surname 2"

    def test_abstract(self, jats_xml_parser):
        abstract = jats_xml_parser.abstract
        assert inspect.isgenerator(abstract)
        abstract = tuple(abstract)
        assert len(abstract) == 2
        assert abstract[0] == "Abstract Paragraph 1"
        assert abstract[1] == "Abstract Paragraph 2"

    def test_paragraphs(self, jats_xml_parser):
        paragraphs = jats_xml_parser.paragraphs
        assert inspect.isgenerator(paragraphs)
        paragraphs = tuple(paragraphs)
        assert len(paragraphs) == 7 + 1 + 2  # for paragraph, caption, table
        # There are 3 caption but one is empty

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

    def test_pubmed_id(self, jats_xml_parser):
        pubmed_id = jats_xml_parser.pubmed_id
        assert isinstance(pubmed_id, str)
        assert pubmed_id == "PMID"

    def test_pmc_id(self, jats_xml_parser):
        pmc_id = jats_xml_parser.pmc_id
        assert isinstance(pmc_id, str)
        assert pmc_id == "PMC"

    def test_doi(self, jats_xml_parser):
        doi = jats_xml_parser.doi
        assert isinstance(doi, str)
        assert doi == "DOI"

    def test_uid(self, jats_xml_parser):
        uid = jats_xml_parser.uid
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
            ("<p>My email is <email>me@epfl.ch</email></p>", "My email is"),
        ),
    )
    def test_inner_text_extraction(
        self, jats_xml_parser, input_xml, expected_inner_text
    ):
        element = ElementTree.fromstring(input_xml)
        inner_text = jats_xml_parser._inner_text(element)
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
            ("<inline-formula>Completely ignored</inline-formula>", ""),
            ("<disp-formula>Block formula</disp-formula>", ""),
            ("<ext-link>https://www.google.com</ext-link>", ""),
            ("<uri>file:///path/to/file</uri>", ""),
            ("<email>me@domain.ai</email>", ""),
            (
                "<unknown-tag>Default: extract inner text.</unknown-tag>",
                "Default: extract inner text.",
            ),
        ),
    )
    def test_element_to_str_works(self, jats_xml_parser, input_xml, expected_str):
        element = ElementTree.fromstring(input_xml)
        element_str = jats_xml_parser._element_to_str(element)
        assert element_str == expected_str

    def test_element_to_str_of_none(self, jats_xml_parser):
        assert jats_xml_parser._element_to_str(None) == ""


@pytest.fixture(scope="session")
def pubmed_xml_parser(test_data_path):
    """Parse a 'PubmedArticle' in a 'PubmedArticleSet'."""
    path = pathlib.Path(test_data_path) / "pubmed_article.xml"
    parser = PubMedXMLParser(path.resolve())
    return parser


@pytest.fixture(scope="session")
def pubmed_xml_parser_minimal(test_data_path):
    """Parse a 'PubmedArticle' in a 'PubmedArticleSet' having only required elements."""
    path = pathlib.Path(test_data_path) / "pubmed_article_minimal.xml"
    parser = PubMedXMLParser(path.resolve())
    return parser


class TestPubMedXMLArticleParser:
    def test_init(self, pubmed_xml_parser):
        assert isinstance(pubmed_xml_parser.content, xml.etree.ElementTree.ElementTree)

    def test_title(self, pubmed_xml_parser):
        title = pubmed_xml_parser.title
        assert title == "Article Title"

    def test_authors(self, pubmed_xml_parser):
        authors = pubmed_xml_parser.authors
        authors = tuple(authors)
        assert len(authors) == 2
        assert authors[0] == "Forenames 1 Lastname 1"
        assert authors[1] == "Lastname 2"

    def test_no_authors(self, pubmed_xml_parser_minimal):
        authors = pubmed_xml_parser_minimal.authors
        authors = tuple(authors)
        assert len(authors) == 0
        assert authors == ()

    def test_abstract(self, pubmed_xml_parser):
        abstract = pubmed_xml_parser.abstract
        abstract = tuple(abstract)
        assert len(abstract) == 2
        assert abstract[0] == "Abstract Paragraph 1"
        assert abstract[1] == "Abstract Paragraph 2"

    def test_no_abstract(self, pubmed_xml_parser_minimal):
        abstract = pubmed_xml_parser_minimal.abstract
        abstract = tuple(abstract)
        assert len(abstract) == 0
        assert abstract == ()

    def test_no_paragraphs(self, pubmed_xml_parser):
        paragraphs = pubmed_xml_parser.paragraphs
        assert len(paragraphs) == 0
        assert paragraphs == ()

    def test_pubmed_id(self, pubmed_xml_parser):
        pubmed_id = pubmed_xml_parser.pubmed_id
        assert isinstance(pubmed_id, str)
        assert pubmed_id == "123456"

    def test_pmc_id(self, pubmed_xml_parser):
        pmc_id = pubmed_xml_parser.pmc_id
        assert isinstance(pmc_id, str)
        assert pmc_id == "PMC12345"

    def test_no_pmc_id(self, pubmed_xml_parser_minimal):
        pmc_id = pubmed_xml_parser_minimal.pmc_id
        assert pmc_id is None

    def test_doi(self, pubmed_xml_parser):
        doi = pubmed_xml_parser.doi
        assert isinstance(doi, str)
        assert doi == "10.0123/issn.0123-4567"

    def test_no_doi(self, pubmed_xml_parser_minimal):
        doi = pubmed_xml_parser_minimal.doi
        assert doi is None

    def test_uid(self, pubmed_xml_parser):
        uid = pubmed_xml_parser.uid
        assert isinstance(uid, str)
        assert uid == "0e8400416a385b9a62d8178539b76daf"


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


class TestTEIXMLArticleParser:
    def test_title(self, tei_xml_parser):
        title = tei_xml_parser.title
        assert isinstance(title, str)
        assert title == "Article Title"

    def test_abstract(self, tei_xml_parser):
        abstract = list(tei_xml_parser.abstract)
        assert len(abstract) == 1
        assert abstract[0] == "Abstract Paragraph 1."

    def test_authors(self, tei_xml_parser):
        authors = list(tei_xml_parser.authors)
        assert len(authors) == 2
        assert authors[0] == "Forename 1 Middle 1 Surname 1"
        assert authors[1] == "Surname 2"

    def test_paragraphs(self, tei_xml_parser):
        paragraphs = list(tei_xml_parser.paragraphs)
        assert len(paragraphs) == 7
        assert paragraphs[0][0] == "Head 1"
        assert paragraphs[2][0] == "Head 2"
        assert paragraphs[4][0] == "Figure Caption"
        assert paragraphs[6][0] == "Table Caption"

        assert paragraphs[0][1] == "Paragraph 1 of Head 1."
        assert paragraphs[3][1] == "Paragraph 2 of (0) Head 2."
        assert paragraphs[4][1] == "Fig. 1. Title."
        assert paragraphs[6][1] == "Table 1. Title."

    def test_no_arxiv_id(self, tei_xml_parser):
        arxiv_id = tei_xml_parser.arxiv_id
        assert arxiv_id is None

    def test_arxiv_id(self, tei_xml_arxiv_parser):
        arxiv_id = tei_xml_arxiv_parser.arxiv_id
        assert isinstance(arxiv_id, str)
        assert arxiv_id == "arxiv:1411.7903v4"

    def test_doi(self, tei_xml_parser):
        doi = tei_xml_parser.doi
        assert isinstance(doi, str)
        assert doi == "DOI 1"

    @pytest.mark.parametrize(
        ("xml_content", "expected_texts"),
        (
            ("", ()),
            ("<p></p>", ()),
            ("<p>Hello.</p>", ("Hello.",)),
            ("<p>Hello</p>", ("Hello.",)),
            ("<p>Hello.</p><p>There.</p>", ("Hello.", "There.")),
            ("<p>Hello</p><p>There.</p>", ("Hello.", "There.")),
            ("<p>Hello</p><p>there.</p>", ("Hello there.",)),
            (
                "<p>This is cool: </p><formula>a + b = c</formula>",
                ("This is cool: FORMULA.",),
            ),
            (
                "<p>As </p><formula>x = 5</formula><p>shows...</p>",
                ("As FORMULA shows...",),
            ),
        ),
    )
    def test_build_texts(self, xml_content, expected_texts, tmp_path):
        tmp_file = tmp_path / "tmp.xml"
        with tmp_file.open("w") as fp:
            fp.write(f"<xml>{xml_content}</xml>")
        parser = TEIXMLParser(tmp_file)
        # Patch the namespace because it's not used in test examples
        parser.tei_namespace["tei"] = ""

        texts = parser._build_texts(parser.content)
        for text, expected_text in zip_longest(texts, expected_texts, fillvalue=None):
            assert text == expected_text

    def test_build_texts_raises_for_unknown_tag(selfs, tmp_path):
        tmp_file = tmp_path / "tmp.xml"
        with tmp_file.open("w") as fp:
            fp.write("<xml><hahaha>HAHAHA</hahaha></xml>")
        parser = TEIXMLParser(tmp_file)
        with pytest.raises(RuntimeError, match=r"Unexpected tag"):
            for _ in parser._build_texts(parser.content):
                # Do nothing, just force the generator to run
                pass


class TestArticle:
    def test_optional_defaults(self):
        article = Article(
            title="",
            authors=("",),
            abstract=("",),
            section_paragraphs=(("", ""),),
        )
        optional_fields = ["pubmed_id", "pmc_id", "doi", "uid"]
        for field in optional_fields:
            assert getattr(article, field) is None

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

from itertools import chain

import pytest

from bluesearch.database.article import Article, ArticleParser, CORD19ArticleParser


@pytest.fixture()
def cord19_json_file():
    json_data = {
        "paper_id": "93140ed88011aaef108208016a0769ad19327dae",
        "metadata": {
            "title": "Some Title",
            "authors": [
                {"first": "Jane", "middle": [], "last": "Doe", "suffix": "Dr."},
                {"first": "Erika", "middle": [], "last": "Mustermann", "suffix": ""},
                {"first": "Jean", "middle": ["X."], "last": "Dupont", "suffix": ""},
                {
                    "first": "Titius",
                    "middle": ["J.", "K."],
                    "last": "Seius",
                    "suffix": "Jr.",
                },
            ],
        },
        "abstract": [
            {
                "section": "Abstract",
                "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            }
        ],
        "body_text": [
            {
                "section": "Introduction",
                "text": "Cras non ante a arcu cursus euismod id a odio.",
            },
            {
                "section": "Introduction",
                "text": "Curabitur eget est vehicula, [...], auctor erat.",
            },
            {
                "section": "Conclusion",
                "text": "Aliquam sit amet lacus vehicula, [...], eleifend odio.",
            },
        ],
        "bib_entries": {},
        "ref_entries": {
            "FIGREF7": {"text": "This figure shows...", "type": "figure"},
            "TABREF23": {
                "text": "The results are presented in the table above.",
                "type": "table",
            },
        },
        "back_matter": [],
    }
    return json_data


class TestParser(ArticleParser):
    def __init__(self):
        self.title = "Test Title"
        self.authors = ("Author 1", "Author 2")
        self.abstract = ("Abstract paragraph 1", "Abstract paragraph 2")
        self.paragraphs = [
            ("Section 1", "Paragraph 1."),
            ("Section 1", "Paragraph 2."),
            ("Section 2", "Paragraph 1."),
        ]

    def get_title(self):
        return self.title

    def iter_authors(self):
        yield from self.authors

    def get_abstract(self):
        return self.abstract

    def iter_paragraphs(self):
        yield from self.paragraphs


class TestCORD19ArticleParser:
    def test_init(self, cord19_json_file):
        # According to the CORD-19 spec the JSON files have to have the
        # keys below.
        parser = CORD19ArticleParser(cord19_json_file)
        assert parser.data == cord19_json_file

        # If any of the key listed above is missing then an exception will
        # be raised.
        with pytest.raises(ValueError, match="Incomplete JSON file"):
            CORD19ArticleParser({})

    def test_get_title(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        title = parser.get_title()
        assert title != ""
        assert title == cord19_json_file["metadata"]["title"]

    def test_iter_authors(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        authors = tuple(parser.iter_authors())

        # Check that all authors have been parsed
        assert len(authors) == len(cord19_json_file["metadata"]["authors"])

        # Check that all name parts of all authors have been collected
        for author, author_dict in zip(
            authors, cord19_json_file["metadata"]["authors"]
        ):
            assert author_dict["first"] in author
            assert author_dict["last"] in author
            assert author_dict["suffix"] in author
            for middle in author_dict["middle"]:
                assert middle in author

    def test_get_abstract(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        abstract = parser.get_abstract()

        # Check that all paragraphs were parsed
        assert len(abstract) == len(cord19_json_file["abstract"])

        # Check that all paragraph texts match
        for paragraph, paragraph_dict in zip(abstract, cord19_json_file["abstract"]):
            assert paragraph == paragraph_dict["text"]

        # Check that if "abstract" is missing then an empty list is returned
        del cord19_json_file["abstract"]
        parser = CORD19ArticleParser(cord19_json_file)
        assert parser.get_abstract() == []

    def test_iter_paragraphs(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        paragraphs = tuple(parser.iter_paragraphs())

        # Check that all paragraphs were parsed
        n_body_text = len(cord19_json_file["body_text"])
        n_ref_entries = len(cord19_json_file["ref_entries"])
        assert len(paragraphs) == n_body_text + n_ref_entries

        # Check that all paragraph texts match
        for (section, text), paragraph_dict in zip(
            paragraphs, cord19_json_file["body_text"]
        ):
            assert section == paragraph_dict["section"]
            assert text == paragraph_dict["text"]

    def test_str(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        parser_str = str(parser)

        # Should be "CORD-19 article ID=<value>" or similar
        assert "CORD-19" in parser_str
        assert str(cord19_json_file["paper_id"]) in parser_str


class TestArticle:
    def test_parse(self):
        # Test article parsing
        parser = TestParser()
        article = Article.parse(parser)
        assert article.title == parser.title
        assert tuple(article.authors) == parser.authors

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
        parser = TestParser()
        article = Article.parse(parser)
        article_str = str(article)
        assert parser.title in article_str
        for author in parser.authors:
            assert author in article_str

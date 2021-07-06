from itertools import chain

import pytest

from bluesearch.database.article import Article, ArticleParser, CORD19ArticleParser


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

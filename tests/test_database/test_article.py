import pytest

from bluesearch.database.article import CORD19ArticleParser


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
        "ref_entries": {},
        "back_matter": [],
    }
    return json_data


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

    def test_iter_paragraphs(self, cord19_json_file):
        parser = CORD19ArticleParser(cord19_json_file)
        paragraphs = tuple(parser.iter_paragraphs())

        # Check that all paragraphs were parsed
        assert len(paragraphs) == len(cord19_json_file["body_text"])

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

    ...

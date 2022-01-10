import datetime
import pathlib

import pytest

import bluesearch
from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo


class TestTopicInfo:
    def test_instantiation(self):
        source = ArticleSource.ARXIV
        path = pathlib.Path("/some/path.test")
        metadata = {"some key": "some value"}
        topic_info = TopicInfo(source, path, metadata=metadata)

        assert topic_info.source == source
        assert topic_info.path == path
        assert metadata == metadata

    @pytest.mark.parametrize(
        ("mapping", "kind", "topics", "result"),
        (
            ({}, "MeSH", ["topic 1"], {"MeSH": ["topic 1"]}),
            (
                {"MeSH": ["topic 2"]},
                "MeSH",
                ["topic 1"],
                {"MeSH": ["topic 1", "topic 2"]},
            ),
            ({"MeSH": ["topic 1"]}, "MeSH", ["topic 1"], {"MeSH": ["topic 1"]}),
        ),
    )
    def test_add_topics(self, mapping, kind, topics, result):
        TopicInfo.add_topics(mapping, kind, topics)
        assert mapping == result

    def test_add_article_journal_topics(self):
        topic_info = TopicInfo(ArticleSource.UNKNOWN, "")
        topic_info.add_article_topics("MeSH", ["AT 1", "AT 2", "AT 3"])
        topic_info.add_journal_topics("MAP", ["JT 1", "JT 2"])

        assert topic_info.article_topics == {"MeSH": ["AT 1", "AT 2", "AT 3"]}
        assert topic_info.journal_topics == {"MAP": ["JT 1", "JT 2"]}

    def test_json(self):
        start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        topic_info = TopicInfo(
            source=ArticleSource.PUBMED,
            path=pathlib.Path("/some/path.test"),
            element_in_file=5,
            metadata={"some key": "some value"},
        )
        topic_info.add_article_topics("MeSH", ["AT 1", "AT 2", "AT 3"])
        topic_info.add_journal_topics("MAP", ["JT 1", "JT 2"])

        end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        json = topic_info.json()
        metadata = json.pop("metadata")
        assert json == {
            "source": ArticleSource.PUBMED.value,
            "path": "/some/path.test",
            "topics": {
                "article": {"MeSH": ["AT 1", "AT 2", "AT 3"]},
                "journal": {"MAP": ["JT 1", "JT 2"]},
            },
        }
        assert start <= metadata["created-date"] <= end
        assert metadata["bbs-version"] == bluesearch.__version__
        assert metadata["some key"] == "some value"

    def test_element_in_file(self):
        json = TopicInfo(ArticleSource.UNKNOWN, "").json()
        assert json["metadata"].get("element_in_file") is None

        json = TopicInfo(ArticleSource.UNKNOWN, "", element_in_file=5).json()
        assert json["metadata"].get("element_in_file") == 5

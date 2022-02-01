from __future__ import annotations

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
        topic_info = TopicInfo(source, path)

        assert topic_info.source == source
        assert topic_info.path == path

    def test_relative_path_is_resolved(self):
        source = ArticleSource.ARXIV
        path = pathlib.Path("relative/path")
        topic_info = TopicInfo(source, path)

        assert topic_info.source == source
        assert topic_info.path == pathlib.Path.cwd() / path

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
        TopicInfo._add_topics(mapping, kind, topics)
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

    def test_element_in_file(self):
        json = TopicInfo(ArticleSource.UNKNOWN, "").json()
        assert json["metadata"].get("element_in_file") is None

        json = TopicInfo(ArticleSource.UNKNOWN, "", element_in_file=5).json()
        assert json["metadata"].get("element_in_file") == 5

    def test_from_dict(self):
        data: dict[str, str | dict] = {
            "source": "pmc",
            "path": "/some/path.test",
            "topics": {
                "article": {"MeSH": ["AT 1", "AT 2", "AT 3"]},
                "journal": {"MAP": ["JT 1", "JT 2"]},
            },
            "metadata": {},
        }
        topic_info = TopicInfo.from_dict(data)
        assert topic_info.source is ArticleSource.PMC
        assert topic_info.path == pathlib.Path("/some/path.test")
        assert isinstance(data["topics"], dict)
        assert topic_info.article_topics == data["topics"]["article"]
        assert topic_info.journal_topics == data["topics"]["journal"]

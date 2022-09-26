from pathlib import Path

import pytest
from elasticsearch import Elasticsearch

from bluesearch.entrypoint.database import add_es


def test(get_es_client: Elasticsearch, tmp_path: Path) -> None:
    from bluesearch.database.article import Article
    from bluesearch.k8s.create_indices import (
        MAPPINGS_ARTICLES,
        MAPPINGS_PARAGRAPHS,
        SETTINGS,
        add_index,
    )

    client = get_es_client
    if client is None:
        pytest.skip("Elastic search is not available")

    article_1 = Article(
        title="some test title",
        authors=["A", "B"],
        abstract=["some test abstract", "abcd"],
        section_paragraphs=[
            ("intro", "some test section_paragraphs 1client"),
            ("summary", "some test section_paragraphs 2"),
        ],
        uid="1",
    )

    article_2 = Article(
        title="SOME test title",
        authors=["Caa adsfaf", "Ddfs fdssf"],
        abstract=["dsaklf", "abcd"],
        section_paragraphs=[
            ("intro", "some TESTTT section_paragraphs 1client"),
            ("summary", "some other test section_paragraphs 2"),
        ],
        uid="2",
    )

    article_1_path = tmp_path / "article_1.json"
    article_2_path = tmp_path / "article_2.json"

    article_1_path.write_text(article_1.to_json())
    article_2_path.write_text(article_2.to_json())

    assert set(client.indices.get_alias().keys()) == set()
    add_index(client, "articles", SETTINGS, MAPPINGS_ARTICLES)
    add_index(client, "paragraphs", SETTINGS, MAPPINGS_PARAGRAPHS)

    add_es.run(client, parsed_path=tmp_path)
    client.indices.refresh(index=["articles", "paragraphs"])

    assert set(client.indices.get_alias().keys()) == {"articles", "paragraphs"}

    # verify articles
    resp = client.search(index="articles", query={"match_all": {}})
    assert resp["hits"]["total"]["value"] == 2

    for doc in resp["hits"]["hits"]:
        if doc["_id"] == "1":
            assert doc["_source"]["abstract"] == ["some test abstract", "abcd"]
            assert doc["_source"]["authors"] == ["A", "B"]
            assert doc["_source"]["title"] == "some test title"
        else:
            assert doc["_source"]["abstract"] == ["dsaklf", "abcd"]
            assert doc["_source"]["authors"] == ["Caa adsfaf", "Ddfs fdssf"]
            assert doc["_source"]["title"] == "SOME test title"

    # verify paragraphs
    resp = client.search(index="paragraphs", query={"match_all": {}})
    assert resp["hits"]["total"]["value"] == 4

    all_docs = set()
    for doc in resp["hits"]["hits"]:
        all_docs.add(
            (
                doc["_source"]["article_id"],
                doc["_source"]["paragraph_id"],
                doc["_source"]["section_name"],
                doc["_source"]["text"],
            )
        )

    all_docs_expected = {
        ("1", 0, "intro", "some test section_paragraphs 1client"),
        ("1", 1, "summary", "some test section_paragraphs 2"),
        ("2", 0, "intro", "some TESTTT section_paragraphs 1client"),
        ("2", 1, "summary", "some other test section_paragraphs 2"),
    }

    assert all_docs == all_docs_expected

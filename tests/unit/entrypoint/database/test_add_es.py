import pytest

from bluesearch.entrypoint.database import add_es


def test(get_es_client, tmp_path):
    from bluesearch.database.article import Article

    client = get_es_client
    if client is None:
        pytest.skip("Elastic search is not available")

    article_1 = Article(
        title="some test title",
        authors="some test authors",
        abstract="some test abstract",
        section_paragraphs=(
            "some test section_paragraphs 1client",
            "some test section_paragraphs 2",
        ),
        uid="some test uid",
    )

    article_2 = Article(
        title="SOME test title",
        authors="SOME test authors",
        abstract="SOME test abstract",
        section_paragraphs=(
            "SOME test section_paragraphs 1",
            "SOME test section_paragraphs 2",
        ),
        uid="SOME test uid",
    )

    article_1_path = tmp_path / "article_1.json"
    article_2_path = tmp_path / "article_2.json"

    article_1_path.write_text(article_1.to_json())
    article_2_path.write_text(article_2.to_json())

    assert set(client.indices.get_alias().keys()) == set()

    add_es.run(parsed_path=tmp_path)
    client.indices.refresh(index=["articles", "paragraphs"])

    assert set(client.indices.get_alias().keys()) == {"articles", "paragraphs"}

    # verify articles
    resp = client.search(index="articles", query={"match_all": {}})
    assert resp["hits"]["total"]["value"] == 2

    # verify paragraphs
    resp = client.search(index="paragraphs", query={"match_all": {}})
    assert resp["hits"]["total"]["value"] == 4

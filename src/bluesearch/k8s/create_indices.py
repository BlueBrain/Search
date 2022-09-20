from __future__ import annotations
import logging
from typing import Any, Optional

from bluesearch.k8s.connect import connect

logger = logging.getLogger(__name__)

SETTINGS = {"number_of_shards": 2, "number_of_replicas": 1}

MAPPINGS_ARTICLES = {
    "dynamic": "strict",
    "properties": {
        "article_id": {"type": "keyword"},
        "doi": {"type": "keyword"},
        "pmc_id": {"type": "keyword"},
        "pubmed_id": {"type": "keyword"},
        "arxiv_id": {"type": "keyword"},
        "title": {"type": "text"},
        "authors": {"type": "text"},
        "abstract": {"type": "text"},
        "journal": {"type": "keyword"},
        "publish_time": {"type": "date", "format": "yyyy-MM-dd"},
        "license": {"type": "keyword"},
        "is_english": {"type": "boolean"},
        "topics": {"type": "keyword"},
    },
}

MAPPINGS_PARAGRAPHS = {
    "dynamic": "strict",
    "properties": {
        "article_id": {"type": "keyword"},
        "section_name": {"type": "keyword"},
        "paragraph_id": {"type": "short"},
        "text": {"type": "text"},
        "is_bad": {"type": "boolean"},
        "embedding": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "dot_product",
        },
    },
}


def add_index(
    index: str,
    settings: dict[str, Any] | None = None,
    mappings: dict[str, Any] | None = None,
) -> None:
    client = connect()

    if index in client.indices.get_alias().keys():
        raise RuntimeError("Index already in ES")

    try:
        client.indices.create(index=index, settings=settings, mappings=mappings)
        logger.info(f"Index {index} created successfully")
    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)


def remove_index(index: str | list[str]) -> None:
    client = connect()

    if index not in client.indices.get_alias().keys():
        raise RuntimeError("Index not in ES")

    try:
        client.indices.delete(index=index)
        logger.info(f"Index {index} deleted successfully")

    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)

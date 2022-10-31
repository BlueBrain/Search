# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""creates an index on ES with the provided name, settings and mappings."""
from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

SETTINGS = {"number_of_shards": 2, "number_of_replicas": 1}

MAPPINGS_ARTICLES: dict[str, Any] = {
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
        "ner_ml": {"type": "flattened"},
        "ner_ruler": {"type": "flattened"},
        "re": {"type": "flattened"},
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
    client: Elasticsearch,
    index: str,
    settings: dict[str, Any] | None = None,
    mappings: dict[str, Any] | None = None,
) -> None:
    """Add an index to ES.

    Parameters
    ----------
    client: Elasticsearch
        Elasticsearch client.
    index: str
        Name of the index.
    settings: dict[str, Any] | None
        Settings of the index.
    mappings: dict[str, Any] | None
        Mappings of the index.

    Returns
    -------
    None
    """
    if index in client.indices.get_alias().keys():
        raise RuntimeError("Index already in ES")

    try:
        client.indices.create(index=index, settings=settings, mappings=mappings)
        logger.info(f"Index {index} created successfully")
    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)


def remove_index(client: Elasticsearch, index: str | list[str]) -> None:
    """Remove an index from ES.

    Parameters
    ----------
    client: Elasticsearch
        Elasticsearch client.
    index: str | list[str]
        Name of the index.

    Returns
    -------
    None
    """
    if index not in client.indices.get_alias().keys():
        raise RuntimeError("Index not in ES")

    try:
        client.indices.delete(index=index)
        logger.info(f"Index {index} deleted successfully")

    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)


def update_index_mapping(
    client: Elasticsearch,
    index: str,
    settings: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    """Update the index with a new mapping and settings."""
    if index not in client.indices.get_alias().keys():
        raise RuntimeError("Index not in ES")

    try:
        if settings:
            client.indices.put_settings(index=index, settings=settings)
        if properties:
            client.indices.put_mapping(index=index, properties=properties)
        logger.info(f"Index {index} updated successfully")
    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)

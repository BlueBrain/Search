import logging
from typing import Any, Optional

from bluesearch.k8s.connect import connect

logger = logging.getLogger(__name__)


def add_index(
    index: str | list[str],
    settings: Optional[dict[str, Any]] = None,
    mappings: Optional[dict[str, Any]] = None,
) -> None:
    client = connect()

    if any(x in client.indices.get_alias().keys() for x in list(index)):
        raise RuntimeError("Index already in ES")

    try:
        client.indices.create(index=index, settings=settings, mappings=mappings)
        logger.info(f"Index {index} created successfully")
    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)


def remove_index(index: str | list[str]) -> None:
    client = connect()

    if not all(x in client.indices.get_alias().keys() for x in list(index)):
        raise RuntimeError("Index not in ES")

    try:
        client.indices.delete(index=index)
        logger.info(f"Index {index} deleted successfully")

    except Exception as err:
        print("Elasticsearch add_index ERROR:", err)

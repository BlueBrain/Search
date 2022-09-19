import logging
from typing import Any

from bluesearch.k8s.connect import connect

logger = logging.getLogger(__name__)


def add_index(index: str, settings: dict[str, Any], mappings: dict[str, Any]) -> None:
    client = connect()

    client.indices.create(index=index, settings=settings, mappings=mappings)

    assert index in client.indices.get_alias().keys(), "index not created"

    logger.info(f"Index {index} created successfully")

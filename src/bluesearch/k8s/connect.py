import logging

import urllib3
from decouple import config
from elasticsearch import Elasticsearch

urllib3.disable_warnings()

logger = logging.getLogger(__name__)


def connect() -> Elasticsearch:
    """return a client connect to BBP K8S"""
    client = Elasticsearch(
        config("ES_URL"),
        basic_auth=("elastic", config("ES_PASS")),
        verify_certs=False,
    )

    if not client.ping():
        raise RuntimeError(f"Cannot connect to BBP K8S: {client.info()}")

    logger.info("Connected to BBP K8S")

    return client


if __name__ == "__main__":
    connect()

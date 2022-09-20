import logging
import os

import urllib3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
urllib3.disable_warnings()

logger = logging.getLogger(__name__)


def connect() -> Elasticsearch:
    """return a client connect to BBP K8S"""
    client = Elasticsearch(
        os.environ["ES_URL"],
        basic_auth=("elastic", os.environ["ES_PASS"]),
        verify_certs=False,
    )

    if not client.ping():
        raise RuntimeError(f"Cannot connect to BBP K8S: {client.info()}")

    logger.info("Connected to BBP K8S")

    return client


if __name__ == "__main__":
    connect()

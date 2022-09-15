import urllib3
from decouple import config
from elasticsearch import Elasticsearch

urllib3.disable_warnings()


def connect():
    """return a client connect to BBP K8S"""
    client = Elasticsearch(
        config("ES_URL"),
        basic_auth=("elastic", config("ES_PASS")),
        verify_certs=False,
    )
    return client

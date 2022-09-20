from decouple import config


def test_es_url():
    assert config("ES_URL")[:4] == "http"

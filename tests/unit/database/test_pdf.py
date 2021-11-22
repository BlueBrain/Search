import re

import pytest
import requests
import responses

from bluesearch.database.pdf import grobid_is_alive, grobid_pdf_to_tei_xml


@responses.activate
def test_conversion_pdf(monkeypatch):
    """Test PDF conversion"""

    responses.add(
        responses.POST,
        "http://fake_host:8888/api/processFulltextDocument",
        body="body",
    )

    result = grobid_pdf_to_tei_xml(b"", host="fake_host", port=8888)
    assert result == "body"
    assert len(responses.calls) == 1


@pytest.mark.parametrize(
    ("body", "expected_result"),
    (
        ("true", True),
        (requests.RequestException(), False),
        ("false", False),
        ("unknown", False),
    ),
)
@responses.activate
def test_grobid_is_alive(body, expected_result):
    host = "host"
    port = 12345
    responses.add(
        responses.GET,
        re.compile(fr"http://{host}:{port}/.*"),
        body=body,
    )
    assert grobid_is_alive(host, port) is expected_result

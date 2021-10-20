import responses

from bluesearch.database.pdf import convert_pdfs_to_tei_xml


@responses.activate
def test_conversion_pdf(monkeypatch):
    """Test PDF conversion"""

    responses.add(
        responses.POST,
        "http://fake_host:8888/api/processFulltextDocument",
        body="body",
    )

    result = convert_pdfs_to_tei_xml(b"", host="fake_host", port="8888")
    assert result == "body"

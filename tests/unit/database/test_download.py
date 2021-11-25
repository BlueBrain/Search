"""Tests for download module."""
from datetime import datetime

import pytest
import responses

from bluesearch.database.download import (
    download_articles,
    get_days_list,
    get_pmc_urls,
)


def test_get_days_list():
    start_date = datetime.strptime("2021-11", "%Y-%m")
    end_date = datetime.strptime("2021-12", "%Y-%m")
    days_list = get_days_list(start_date, end_date)
    assert isinstance(days_list, list)
    assert len(days_list) == 31  # 30 days in November + 1 in December
    for day in days_list:
        assert isinstance(day, datetime)

    days_list = get_days_list(start_date)
    assert isinstance(days_list, list)
    for day in days_list:
        assert isinstance(day, datetime)


@pytest.mark.parametrize(
    ("component", "expected_url_start"),
    [
        ("author_manuscript", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml/"),
        ("oa_comm", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"),
        ("oa_noncomm", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/"),
    ],
)
def test_get_pmc_urls(monkeypatch, component, expected_url_start):

    start_date = datetime.strptime("2021-11", "%Y-%m")
    end_date = datetime(2021, 11, 23, 0, 0, 0, 0)

    url_list = get_pmc_urls(component, start_date, end_date=end_date)
    assert isinstance(url_list, list)
    assert len(url_list) == 23
    for url in url_list:
        assert url.startswith(expected_url_start)


@responses.activate
def test_download_articles(tmp_path):
    path_names = ["file1.txt", "file2.txt"]
    url_list = []
    for name in path_names:
        url = "http://fake/url/" + name
        url_list.append(url)
        responses.add(responses.GET, url, body="fake string")

    download_articles(url_list, tmp_path)
    assert len(list(tmp_path.iterdir())) == 2
    for file in tmp_path.iterdir():
        assert file.name in path_names

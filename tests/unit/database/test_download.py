"""Tests for download module."""
from datetime import datetime

import pytest
import responses

from bluesearch.database.download import (
    download_articles,
    download_articles_s3,
    generate_pmc_urls,
    get_daterange_list,
    get_pubmed_urls,
    get_s3_urls,
)


class TestGetDaterangeList:
    def test_delta_day(self):
        start_date = datetime.strptime("2021-11", "%Y-%m")
        end_date = datetime.strptime("2021-12", "%Y-%m")
        days_list = get_daterange_list(start_date, end_date, delta="day")
        assert isinstance(days_list, list)
        assert len(days_list) == 31  # 30 days in November + 1 in December
        for day in days_list:
            assert isinstance(day, datetime)

        days_list = get_daterange_list(start_date, delta="day")
        assert isinstance(days_list, list)
        for day in days_list:
            assert isinstance(day, datetime)

    def test_delta_month(self):
        start_date = datetime.strptime("2020-11", "%Y-%m")
        end_date = datetime.strptime("2021-11", "%Y-%m")

        months_list = get_daterange_list(start_date, end_date, delta="month")
        assert isinstance(months_list, list)
        assert len(months_list) == 13

        assert all(date.day == 1 for date in months_list)

        months_indices = [date.month for date in months_list]

        assert months_indices == [
            11,
            12,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]

    def test_delta_wrong(self):
        with pytest.raises(ValueError, match="Unknown delta"):
            get_daterange_list(datetime.today(), delta="wrong delta")



@pytest.mark.parametrize(
    ("component", "expected_url_start"),
    [
        ("author_manuscript", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml/"),
        ("oa_comm", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"),
        ("oa_noncomm", "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/"),
    ],
)
def test_generate_pmc_urls(monkeypatch, component, expected_url_start):

    start_date = datetime.strptime("2021-11", "%Y-%m")
    end_date = datetime(2021, 11, 23, 0, 0, 0, 0)

    url_list = generate_pmc_urls(component, start_date, end_date=end_date)
    assert isinstance(url_list, list)
    assert len(url_list) == 23
    for url in url_list:
        assert url.startswith(expected_url_start)


@responses.activate
def test_get_pubmed_urls(monkeypatch, test_data_path):
    html_path = test_data_path / "pubmed_download_index.html"
    assert html_path.exists()

    source_code = html_path.read_text()
    expected_url_start = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"
    responses.add(
        responses.GET,
        expected_url_start,
        body=source_code,
    )

    start_date = datetime.strptime("2020-10", "%Y-%m")
    end_date = datetime(2020, 12, 16, 0, 0, 0, 0)

    url_list = get_pubmed_urls(start_date, end_date=end_date)
    assert isinstance(url_list, list)
    assert len(url_list) == 6  # we counted it manually in html_path
    for url in url_list:
        assert url.startswith(expected_url_start)

def test_get_s3_urls():
    pass


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

def test_download_articles_s3():
    pass

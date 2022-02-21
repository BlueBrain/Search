"""Tests for download module."""
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import responses
from google.cloud.storage import Blob, Bucket

from bluesearch.database.download import (
    download_articles,
    download_gcs_blob,
    download_s3_articles,
    generate_pmc_urls,
    get_daterange_list,
    get_gcs_urls,
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
    n_papers_per_month = 20
    S3object = namedtuple("S3object", ["key"])

    fake_bucket = Mock()
    return_value = [S3object("whatever.meca") for _ in range(n_papers_per_month)] + [
        S3object("some_folder/")
    ]
    fake_bucket.objects.filter.return_value = return_value

    start_date = datetime(2019, 11, 13)
    end_date = datetime(2020, 2, 22)

    url_dict = get_s3_urls(fake_bucket, start_date, end_date)

    expected_keys = {
        "November_2019",
        "December_2019",
        "January_2020",
        "February_2020",
    }

    assert isinstance(url_dict, dict)
    assert set(url_dict.keys()) == expected_keys

    for _, url_list in url_dict.items():
        assert len(url_list) == n_papers_per_month


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


def test_download_s3_articles(tmp_path):
    def fake_download_file(Key, Filename, ExtraArgs):
        Path(Filename).touch()

    fake_bucket = Mock()
    fake_bucket.download_file.side_effect = fake_download_file

    url_dict = {
        "November_2018": ["Current_Content/November_2018/1.meca"],
        "December_2018": ["Current_Content/December_2018/2.meca"],
        "January_2019": [
            "Current_Content/January_2019/3.meca",
            "Current_Content/January_2019/4.meca",
        ],
    }

    download_s3_articles(fake_bucket, url_dict, tmp_path)

    assert (tmp_path / "Current_Content" / "November_2018" / "1.meca").exists()
    assert (tmp_path / "Current_Content" / "December_2018" / "2.meca").exists()
    assert (tmp_path / "Current_Content" / "January_2019" / "3.meca").exists()
    assert (tmp_path / "Current_Content" / "January_2019" / "4.meca").exists()


def test_get_gcs_urls():
    fake_client = Mock()
    fake_bucket = Bucket(fake_client, "my_dir/file.txt")
    fake_blobs_by_prefix = {
        "arxiv/arxiv/pdf/2109": [
            Blob("topic-a/99.6767v1.1.pdf", fake_bucket),  # invalid version
            Blob("topic-v/99.6767v1.2.pdf", fake_bucket),  # invalid version
            Blob("topic-v/99.6767v1a.pdf", fake_bucket),  # invalid version
            Blob("topic-v/99.6767v10.pdf", fake_bucket),
            Blob("topic-v/99.6767v3.pdf", fake_bucket),  # older version
        ],
        "arxiv/arxiv/pdf/2110": [
            Blob("topic-a/12.3450v1.pdf", fake_bucket),  # older version
            Blob("topic-v/12.3450v2.pdf", fake_bucket),
        ],
        "arxiv/arxiv/pdf/2111": [
            Blob("topic-v/99.3450v2.pdf", fake_bucket),  # older version
            Blob("topic-v/99.3450v3.pdf", fake_bucket),  # older version
            Blob("topic-v/99.3450v10.pdf", fake_bucket),
        ],
        "arxiv/arxiv/pdf/2112": [
            Blob("topic-v/33.1v2.pdf", fake_bucket),
            Blob("topic-v/44.1v2.pdf", fake_bucket),
            Blob("topic-v/55.1v2.pdf", fake_bucket),
            Blob("topic-v/55.1v1.pdf", fake_bucket),  # older version
        ],
    }

    fake_client.list_blobs.side_effect = lambda bucket, prefix: fake_blobs_by_prefix[
        prefix
    ]
    start_date = datetime(2021, 9, 1)
    end_date = datetime(2021, 12, 1)
    blobs_by_month = get_gcs_urls(fake_bucket, start_date, end_date)

    assert fake_client.list_blobs.call_count == 4
    assert set(blobs_by_month) == {"2109", "2110", "2111", "2112"}
    assert set(blobs_by_month["2109"]) == set(
        fake_blobs_by_prefix["arxiv/arxiv/pdf/2109"][-2:-1]
    )
    assert set(blobs_by_month["2110"]) == set(
        fake_blobs_by_prefix["arxiv/arxiv/pdf/2110"]
    )
    assert set(blobs_by_month["2111"]) == set(
        fake_blobs_by_prefix["arxiv/arxiv/pdf/2111"][-1:]
    )
    assert set(blobs_by_month["2112"]) == set(
        fake_blobs_by_prefix["arxiv/arxiv/pdf/2112"][:-1]
    )


@pytest.mark.parametrize(
    ("flatten", "name", "path"),
    (
        (False, "subdir/file.pdf", "subdir/file.pdf"),
        (True, "subdir/file.pdf", "file.pdf"),
    ),
)
def test_download_gcs_blob(flatten, name, path, tmp_path):
    blob = Mock()
    blob.name = name
    blob.download_as_bytes.return_value = b"PDF content"

    download_gcs_blob(blob, tmp_path, flatten=flatten)

    pdf_path = tmp_path / path
    assert pdf_path.exists()
    with pdf_path.open("rb") as fh:
        assert fh.read() == b"PDF content"

import argparse
import inspect
import pathlib
from datetime import datetime
from unittest.mock import Mock

import pytest

from bluesearch.entrypoint.database import download

DOWNLOAD_PARAMS = {"source", "from_month", "output_dir", "dry_run"}


def test_init_parser():
    parser = download.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["pmc", "2020-10", "/path/to/download"])
    assert vars(args).keys() == DOWNLOAD_PARAMS

    # Test the values
    assert args.source == "pmc"
    assert args.from_month == datetime(2020, 10, 1)
    assert args.output_dir == pathlib.Path("/path/to/download")
    assert not args.dry_run

    # Invalid date
    parser = download.init_parser(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["pmc", "invalid-date", "/path/to/download"])


def test_run_arguments():
    assert inspect.signature(download.run).parameters.keys() == DOWNLOAD_PARAMS


class TestRun:
    def test_pmc_download(self, capsys, monkeypatch, tmp_path):
        def fake_download_articles_func(url_list, output_dir):
            for url in url_list:
                path = output_dir / url
                path.touch()

        fake_download_articles = Mock(side_effect=fake_download_articles_func)

        monkeypatch.setattr(
            "bluesearch.database.download.download_articles",
            fake_download_articles,
        )

        fake_get_pmc_urls = Mock(return_value=["fake1", "fake2"])
        monkeypatch.setattr(
            "bluesearch.database.download.get_pmc_urls", fake_get_pmc_urls
        )

        fake_datetime = datetime(2021, 11, 1)
        pmc_path = tmp_path / "pmc"
        download.run("pmc", fake_datetime, pmc_path, dry_run=False)
        assert pmc_path.exists()
        assert {path.name for path in pmc_path.iterdir()} == {
            "author_manuscript",
            "oa_comm",
            "oa_noncomm",
        }
        for sub_dir in pmc_path.iterdir():
            assert len(list(sub_dir.iterdir())) == 2
        assert fake_download_articles.call_count == 3
        assert fake_get_pmc_urls.call_count == 3

        fake_get_pmc_urls.reset_mock()
        fake_download_articles.reset_mock()
        download.run("pmc", fake_datetime, pmc_path, dry_run=True)
        captured = capsys.readouterr()
        assert len(captured.out.split("\n")) == 10
        assert fake_get_pmc_urls.call_count == 3
        assert fake_download_articles.call_count == 0

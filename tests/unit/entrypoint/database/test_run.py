# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import inspect
import pathlib

import pytest

from bluesearch.entrypoint.database import run

RUN_PARAMS = {
    "source",
    "from_month",
    "filter_config",
    "output_dir",
    "db_url",
    "db_type",
    "mesh_topic_db",
    "dry_run",
    "grobid_host",
    "grobid_port",
    "identifier",
}


def test_init_parser():
    parser = run.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(
        [
            "--source=arxiv",
            "--from-month=2021-12",
            "--filter-config=/path/to/config.jsonl",
            "--output-dir=some/output/dir",
            "--db-url=some.url",
        ]
    )
    assert vars(args).keys() == RUN_PARAMS

    # Test the values
    assert args.source == "arxiv"
    assert args.from_month == "2021-12"
    assert args.filter_config == pathlib.Path("/path/to/config.jsonl")


def test_run_arguments():
    assert inspect.signature(run.run).parameters.keys() == RUN_PARAMS


@pytest.mark.parametrize(
    "source,tasks",
    [
        (
            "arxiv",
            (
                "DownloadTask",
                "TopicExtractTask",
                "TopicFilterTask",
                "PerformFilteringTask",
                "ConvertPDFTask",
                "ParseTask",
                "AddTask",
            ),
        ),
        (
            "biorxiv",
            (
                "DownloadTask",
                "TopicExtractTask",
                "TopicFilterTask",
                "PerformFilteringTask",
                "ParseTask",
                "AddTask",
            ),
        ),
        (
            "medrxiv",
            (
                "DownloadTask",
                "TopicExtractTask",
                "TopicFilterTask",
                "PerformFilteringTask",
                "ParseTask",
                "AddTask",
            ),
        ),
        (
            "pmc",
            (
                "DownloadTask",
                "UnzipTask",
                "TopicExtractTask",
                "TopicFilterTask",
                "PerformFilteringTask",
                "ParseTask",
                "AddTask",
            ),
        ),
        (
            "pubmed",
            (
                "DownloadTask",
                "TopicExtractTask",
                "TopicFilterTask",
                "PerformFilteringTask",
                "ParseTask",
                "AddTask",
            ),
        ),
    ],
)
def test_pipelines(source, tasks, tmp_path, capsys):
    run.run(
        source=source,
        from_month="whatever",
        filter_config=pathlib.Path("whatever"),
        output_dir=tmp_path,
        dry_run=True,
        mesh_topic_db=pathlib.Path("whatever"),
        grobid_host="whatever",
        grobid_port=1234,
        db_url="whatever",
        db_type="sqlite",
        identifier=None,
    )

    captured = capsys.readouterr()
    stdout_lines = reversed(captured.out.splitlines()[1:])

    for stdout_line, task in zip(stdout_lines, tasks):
        assert task in stdout_line
        assert "PENDING" in stdout_line

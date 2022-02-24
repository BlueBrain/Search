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
from subprocess import Popen
from unittest.mock import Mock

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
    "final_task",
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
        final_task=None,
    )

    captured = capsys.readouterr()
    stdout_lines = reversed(captured.out.splitlines()[1:])

    for stdout_line, task in zip(stdout_lines, tasks):
        assert task in stdout_line
        assert "PENDING" in stdout_line


@pytest.mark.parametrize(
    "source",
    [
        "arxiv",
        "biorxiv",
        "medrxiv",
        "pmc",
        "pubmed",
    ],
)
def test_all(
    tmp_path,
    monkeypatch,
    source,
):
    identifier = "ABC"
    root_dir = tmp_path / source / identifier

    fake_Popen_inst = Mock(spec=Popen)
    fake_Popen_inst.returncode = 0

    def create_output(args, **kwargs):
        entrypoint = args[1]

        if entrypoint == "download":
            output_path = root_dir / "raw/"
            output_path.mkdir(parents=True)

        elif entrypoint == "topic-extract":
            output_path = root_dir / "topic_infos.jsonl"
            output_path.touch()

        elif entrypoint == "topic-filter":
            output_path = root_dir / "filtering.csv"
            output_path.touch()

        elif entrypoint == "convert-pdf":
            output_path = root_dir / "converted_pdfs/"
            output_path.mkdir()

        elif entrypoint == "parse":
            output_path = root_dir / "parsed/"
            output_path.mkdir()

        elif entrypoint == "add":
            pass

        return fake_Popen_inst

    fake_Popen_class = Mock(side_effect=create_output)
    monkeypatch.setattr("subprocess.Popen", fake_Popen_class)
    monkeypatch.setattr(
        run.UnzipTask, "run", lambda _: (root_dir / "raw_unzipped").mkdir()
    )
    monkeypatch.setattr(
        run.PerformFilteringTask, "run", lambda _: (root_dir / "filtered/").mkdir()
    )
    monkeypatch.setattr(run.AddTask, "complete", lambda _: False)

    run.run(
        source=source,
        from_month="1234-11",
        filter_config=pathlib.Path("aa"),
        output_dir=tmp_path,
        dry_run=False,
        mesh_topic_db=pathlib.Path("whatever"),
        grobid_host="112431321",
        grobid_port=8000,
        db_url="whatever",
        db_type="sqlite",
        identifier=identifier,
        final_task="AddTask",
    )
    assert (root_dir / "raw").exists()
    if source == "pmc":
        assert (root_dir / "raw_unzipped").exists()

    assert (root_dir / "topic_infos.jsonl").exists()
    assert (root_dir / "filtering.csv").exists()
    assert (root_dir / "filtered").exists()

    if source == "arxiv":
        assert (root_dir / "converted_pdfs").exists()

    assert (root_dir / "parsed").exists()

    if source == "arxiv":
        assert fake_Popen_class.call_count == 6
    else:
        assert fake_Popen_class.call_count == 5

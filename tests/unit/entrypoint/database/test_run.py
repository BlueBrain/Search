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
from subprocess import Popen
from unittest.mock import Mock

import pytest

from bluesearch.entrypoint.database import run

RUN_PARAMS = {
    "final_task",
    "luigi_config_path",
    "luigi_config_args",
    "dry_run",
}


def test_init_parser():
    parser = run.init_parser(argparse.ArgumentParser())

    args = parser.parse_args([])
    assert vars(args).keys() == RUN_PARAMS

    # # Test the values
    assert args.final_task is None
    assert args.luigi_config_args is None
    assert args.dry_run is False
    assert args.luigi_config_path is None


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
        luigi_config_args=f"GlobalParams.source:{source},"
        f"DownloadTask.output_dir:{tmp_path}",
        dry_run=True,
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
        luigi_config_args=f"GlobalParams.source:{source},"
        f"DownloadTask.output_dir:{tmp_path},"
        f"DownloadTask.identifier:{identifier}",
        dry_run=False,
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

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
import logging
import pathlib
from unittest.mock import Mock

from bluesearch.entrypoint.database import topic_extract
from bluesearch.utils import JSONL

TOPIC_EXTRACT_PARAMS = {
    "source",
    "input_path",
    "output_file",
    "match_filename",
    "recursive",
    "overwrite",
    "dry_run",
}


def test_init_parser():
    parser = topic_extract.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["pmc", "/path/to/input", "/path/to/output"])
    assert vars(args).keys() == TOPIC_EXTRACT_PARAMS

    # Test the values
    assert args.source == "pmc"
    assert args.input_path == pathlib.Path("/path/to/input")
    assert args.output_file == pathlib.Path("/path/to/output")
    assert args.match_filename is None
    assert not args.recursive
    assert not args.overwrite
    assert not args.dry_run


def test_run_arguments():
    assert (
        inspect.signature(topic_extract.run).parameters.keys() == TOPIC_EXTRACT_PARAMS
    )


def test_input_path_not_correct(caplog):
    with caplog.at_level(logging.ERROR):
        exit_code = topic_extract.run(
            source="pmc",
            input_path=pathlib.Path("wrong_directory/"),
            output_file=pathlib.Path(""),
            match_filename=None,
            recursive=False,
            overwrite=False,
            dry_run=False,
        )
    assert exit_code == 1
    assert "Argument 'input_path'" in caplog.text


def test_wrong_source(test_data_path, caplog, tmp_path):
    pmc_path = test_data_path / "jats_article.xml"
    with caplog.at_level(logging.ERROR):
        exit_code = topic_extract.run(
            source="wrong_type",
            input_path=pmc_path,
            output_file=tmp_path,
            match_filename=None,
            recursive=False,
            overwrite=False,
            dry_run=False,
        )
    assert exit_code == 1
    assert "Unknown article source" in caplog.text


def test_dry_run(test_data_path, capsys, tmp_path):
    pmc_path = test_data_path / "jats_article.xml"
    exit_code = topic_extract.run(
        source="pmc",
        input_path=pmc_path,
        output_file=tmp_path,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=True,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(pmc_path) in captured.out


def test_pmc_source(test_data_path, capsys, monkeypatch, tmp_path):
    pmc_path = test_data_path / "jats_article.xml"
    output_jsonl = tmp_path / "test.jsonl"
    meshes = ["MeSH 1", "MeSH 2"]

    get_topic_for_pmc_mock = Mock(return_value=meshes)
    monkeypatch.setattr(
        "bluesearch.database.topic.get_topics_for_pmc_article", get_topic_for_pmc_mock
    )

    exit_code = topic_extract.run(
        source="pmc",
        input_path=pmc_path,
        output_file=output_jsonl,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
    )
    assert exit_code == 0
    assert output_jsonl.exists()
    results = JSONL.load_jsonl(output_jsonl)

    assert len(results) == 1
    result = results[0]
    assert result["source"] == "pmc"
    assert result["path"] == str(pmc_path)
    assert isinstance(result["topics"], dict)
    topics = result["topics"]
    assert "journal" in topics
    assert isinstance(topics["journal"], dict)
    assert topics["journal"]["MeSH"] == meshes
    assert "metadata" in result

    # Test overwrite
    exit_code = topic_extract.run(
        source="pmc",
        input_path=pmc_path,
        output_file=output_jsonl,
        match_filename=None,
        recursive=False,
        overwrite=True,
        dry_run=False,
    )
    assert exit_code == 0
    results = JSONL.load_jsonl(output_jsonl)
    assert len(results) == 1  # Length still 1 because we overwrite

    # Test appending
    exit_code = topic_extract.run(
        source="pmc",
        input_path=pmc_path,
        output_file=output_jsonl,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
    )
    assert exit_code == 0
    results = JSONL.load_jsonl(output_jsonl)
    assert len(results) == 2  # Length 2 because we append the file


def test_pubmed_source(test_data_path, capsys, monkeypatch, tmp_path):
    pmc_path = test_data_path / "pubmed_articles.xml"
    output_jsonl = tmp_path / "test.jsonl"
    journal_meshes = ["MeSH Journal 1", "MeSH Journal 2"]
    article_meshes = ["MeSH Article 1", "MeSH Article 2"]

    extract_article_topic_for_pubmed_mock = Mock(return_value=article_meshes)
    monkeypatch.setattr(
        "bluesearch.database.topic.extract_article_topics_for_pubmed_article",
        extract_article_topic_for_pubmed_mock,
    )

    extract_journal_topic_for_pubmed_mock = Mock(return_value=journal_meshes)
    monkeypatch.setattr(
        "bluesearch.database.topic.extract_journal_topics_for_pubmed_article",
        extract_journal_topic_for_pubmed_mock,
    )

    exit_code = topic_extract.run(
        source="pubmed",
        input_path=pmc_path,
        output_file=output_jsonl,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
    )
    assert exit_code == 0
    assert output_jsonl.exists()
    results = JSONL.load_jsonl(output_jsonl)
    assert extract_journal_topic_for_pubmed_mock.call_count == 2
    assert extract_article_topic_for_pubmed_mock.call_count == 2

    assert len(results) == 2
    result = results[0]
    assert result["source"] == "pubmed"
    assert result["path"] == str(pmc_path)
    assert isinstance(result["topics"], dict)
    topics = result["topics"]
    assert "article" in topics
    assert "journal" in topics
    assert isinstance(topics["journal"], dict)
    assert isinstance(topics["article"], dict)
    assert topics["journal"]["MeSH"] == journal_meshes
    assert topics["article"]["MeSH"] == article_meshes
    assert "metadata" in result
    assert "element_in_file" in result["metadata"]

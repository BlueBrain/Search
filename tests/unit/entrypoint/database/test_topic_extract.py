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
import gzip
import inspect
import json
import logging
import pathlib
from unittest.mock import Mock

import pytest

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
    "mesh_topic_db",
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


def test_source_type_not_implemented(test_data_path, caplog, tmp_path):
    pmc_path = test_data_path / "jats_article.xml"
    with caplog.at_level(logging.ERROR):
        exit_code = topic_extract.run(
            source="unknown",
            input_path=pmc_path,
            output_file=tmp_path,
            match_filename=None,
            recursive=False,
            overwrite=False,
            dry_run=False,
        )
    assert exit_code == 1
    assert "not implemented" in caplog.text


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
    mesh_tree_path = tmp_path / "mesh_tree.json"
    mesh_tree_numbers = {
        "A1": "topic1",
        "A1.1": "topic11",
        "A1.2": "topic12",
        "A1.2.1": "topic121",
    }
    mesh_tree_path.write_text(json.dumps(mesh_tree_numbers))
    meshes = ["topic11", "topic12"]

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
        mesh_topic_db=mesh_tree_path,
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
    assert topics["journal"]["MeSH"] == ["topic1", "topic11", "topic12"]
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
        mesh_topic_db=mesh_tree_path,
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
        mesh_topic_db=mesh_tree_path,
    )
    assert exit_code == 0
    results = JSONL.load_jsonl(output_jsonl)
    assert len(results) == 2  # Length 2 because we append the file


@pytest.mark.parametrize("source", ["biorxiv", "medrxiv"])
def test_medbiorxiv_source(capsys, monkeypatch, tmp_path, source):
    input_path = tmp_path / "1234.xml"
    output_file = tmp_path / "output.jsonl"
    input_path.touch()

    # Mocking
    fake_extract_article_topics_from_medrxiv_article = Mock(
        side_effect=lambda p: ("TOPIC", source)
    )

    monkeypatch.setattr(
        "bluesearch.database.topic.extract_article_topics_from_medrxiv_article",
        fake_extract_article_topics_from_medrxiv_article,
    )

    topic_extract.run(
        source=source,
        input_path=input_path,
        output_file=output_file,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
    )

    assert output_file.exists()
    fake_extract_article_topics_from_medrxiv_article.assert_called_once()

    result = JSONL.load_jsonl(output_file)
    assert len(result) == 1

    assert result[0]["source"] == source
    assert result[0]["topics"]["article"]["Subject Area"] == ["TOPIC"]


def test_pubmed_source(test_data_path, pubmed_articles_zipped_path, capsys, monkeypatch, tmp_path):

    mesh_tree_path = tmp_path / "mesh_tree.json"

    output_jsonl = tmp_path / "test.jsonl"
    mesh_tree_numbers = {
        "A1": "topic1",
        "A1.1": "topic11",
        "A1.2": "topic12",
        "A1.2.1": "topic121",
    }
    mesh_tree_path.write_text(json.dumps(mesh_tree_numbers))
    journal_meshes = ["topic11", "topic12"]
    article_meshes = ["topic11", "topic121"]
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
        input_path=pubmed_articles_zipped_path,
        output_file=output_jsonl,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
        mesh_topic_db=mesh_tree_path,
    )
    assert exit_code == 0
    assert output_jsonl.exists()
    results = JSONL.load_jsonl(output_jsonl)
    assert extract_journal_topic_for_pubmed_mock.call_count == 2
    assert extract_article_topic_for_pubmed_mock.call_count == 2

    assert len(results) == 2
    result = results[0]
    assert result["source"] == "pubmed"
    assert result["path"] == str(pubmed_articles_zipped_path)
    assert isinstance(result["topics"], dict)
    topics = result["topics"]
    assert "article" in topics
    assert "journal" in topics
    assert isinstance(topics["journal"], dict)
    assert isinstance(topics["article"], dict)
    assert topics["journal"]["MeSH"] == ["topic1", "topic11", "topic12"]
    assert topics["article"]["MeSH"] == ["topic1", "topic11", "topic12", "topic121"]
    assert "metadata" in result
    assert "element_in_file" in result["metadata"]


@pytest.mark.parametrize("source", ["pubmed", "pmc"])
def test_mesh_topic_db_is_enforced(source, caplog, tmp_path):
    exit_code = topic_extract.run(
        source=source,
        input_path=tmp_path,
        output_file=tmp_path,
        match_filename=None,
        recursive=False,
        overwrite=False,
        dry_run=False,
    )
    assert exit_code != 0
    assert "--mesh-topics-db is mandatory" in caplog.text

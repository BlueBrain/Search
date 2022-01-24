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
import re

import pandas as pd
import pytest

from bluesearch.database.article import ArticleSource
from bluesearch.database.topic_info import TopicInfo
from bluesearch.entrypoint.database import topic_filter
from bluesearch.entrypoint.database.topic_filter import TopicRule
from bluesearch.utils import JSONL

TOPIC_FILTER_PARAMS = {
    "extracted_topics",
    "filter_config",
    "output_file",
}


def test_init_parser():
    parser = topic_filter.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["/path/to/topics", "/path/to/config", "/path/to/output"])
    assert vars(args).keys() == TOPIC_FILTER_PARAMS

    # Test the values
    assert args.extracted_topics == pathlib.Path("/path/to/topics")
    assert args.filter_config == pathlib.Path("/path/to/config")
    assert args.output_file == pathlib.Path("/path/to/output")


def test_run_arguments():
    assert inspect.signature(topic_filter.run).parameters.keys() == TOPIC_FILTER_PARAMS


EXTRACTED_TOPICS_EXAMPLE = [
    # bioarxiv
    {
        "metadata": {
            "bbs-version": "1.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_bioarxiv_1",
        "source": "biorxiv",
        "topics": {"article": {"Subject Area": ["Neuroscience"]}, "journal": {}},
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_bioarxiv_2",
        "source": "biorxiv",
        "topics": {"article": {"Subject Area": ["Microbiology"]}, "journal": {}},
    },
    # medrxiv
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_medrxiv_1",
        "source": "medrxiv",
        "topics": {"article": {"Subject Area": ["Epidemiology"]}, "journal": {}},
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_medrxiv_2",
        "source": "medrxiv",
        "topics": {
            "article": {"Subject Area": ["Psychiatry and Clinical Psychology"]},
            "journal": {},
        },
    },
    # arxiv
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:30:42",
        },
        "path": "path_arxiv_1",
        "source": "arxiv",
        "topics": {
            "article": {"arXiv": ["nlin.PS", "physics.chem-ph", "q-bio.MN"]},
            "journal": {},
        },
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:30:42",
        },
        "path": "path_arxiv_2",
        "source": "arxiv",
        "topics": {
            "article": {"arXiv": ["math.CA", "math.PR", "math.ST", "stat.TH"]},
            "journal": {},
        },
    },
    # pmc
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:39:41",
        },
        "path": "path_pmc_1",
        "source": "pmc",
        "topics": {"article": {}, "journal": {"MeSH": ["Neoplasms"]}},
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:39:41",
        },
        "path": "path_pmc_2",
        "source": "pmc",
        "topics": {
            "article": {},
            "journal": {
                "MeSH": [
                    "Biomedical Research",
                    "Biotechnology",
                    "Microfluidic Analytical Techniques",
                    "Molecular Biology",
                ]
            },
        },
    },
    # pubmed
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:53:00",
            "element_in_file": 386,
        },
        "path": "path_pubmed_1",
        "source": "pubmed",
        "topics": {
            "article": {
                "MeSH": [
                    "Humans",
                    "Physical Examination",
                    "Quality of Life",
                    "Skin Care",
                ]
            },
            "journal": {"MeSH": ["Nursing", "United Kingdom"]},
        },
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-20 11:53:00",
            "element_in_file": 387,
        },
        "path": "path_pubmed_2",
        "source": "pubmed",
        "topics": {
            "article": {
                "MeSH": [
                    "3' Untranslated Regions",
                    "Animals",
                    "Avoidance Learning",
                    "CA1 Region, Hippocampal",
                    "Gene Expression Regulation",
                    "Injections, Intraventricular",
                    "Isoenzymes",
                    "Long-Term Potentiation",
                    "Male",
                    "Maze Learning",
                    "Mice",
                    "Mice, Inbred C57BL",
                    "Neuronal Plasticity",
                    "Obsessive Behavior",
                    "Polynucleotide Adenylyltransferase",
                    "Protein Biosynthesis",
                    "RNA Processing, Post-Transcriptional",
                    "RNA, Messenger",
                    "RNA, Small Interfering",
                    "Transcription Factors",
                    "Transcription, Genetic",
                    "mRNA Cleavage and Polyadenylation Factors",
                ]
            },
            "journal": {"MeSH": ["Molecular Biology", "RNA"]},
        },
    },
]

class TestTopicRule:
    def test_noparams(self):
        rule = TopicRule()

        assert rule.level is None
        assert rule.source is None
        assert rule.pattern is None

    def test_level_validation(self):
        rule_1 = TopicRule(level="article")
        rule_2 = TopicRule(level="journal")

        assert rule_1.level == "article"
        assert rule_2.level == "journal"

        with pytest.raises(ValueError, match="Unsupported level"):
            TopicRule(level="wrong")

    def test_source_validation(self):
        rule_1 = TopicRule(source="arxiv")
        rule_2 = TopicRule(source=ArticleSource("biorxiv"))
        rule_3 = TopicRule(source=ArticleSource.PUBMED)

        assert rule_1.source is ArticleSource.ARXIV
        assert rule_2.source is ArticleSource.BIORXIV
        assert rule_3.source is ArticleSource.PUBMED

        with pytest.raises(ValueError, match="Unsupported source"):
            TopicRule(source="wrong_source")

    def test_pattern_validation(self):
        rule_1 = TopicRule(pattern="some_pattern")
        rule_2 = TopicRule(pattern=re.compile("whatever"))

        assert rule_1.pattern.pattern == "some_pattern"
        assert rule_2.pattern.pattern == "whatever"

        with pytest.raises(ValueError, match="Unsupported pattern"):
            TopicRule(pattern=r"\x")

    def test_matching(self):
        info = TopicInfo.from_dict(
            {
                "source": "arxiv",
                "path": "some_path",
                "topics": {
                    "article": {
                        "some_key": ["book", "food"],
                        "some_other_key": ["meat"],
                    },
                    "journal": {
                        "some_key": ["pasta"],
                    },
                },
            }
        )

        rule_1 = TopicRule(pattern="oo")
        rule_2 = TopicRule()
        rule_3 = TopicRule(level="journal", pattern="asta")
        rule_4 = TopicRule(source="biorxiv")
        rule_5 = TopicRule(level="article", pattern="eat")
        rule_6 = TopicRule(level="article", pattern="eataaa")
        rule_7 = TopicRule(level="journal", pattern="837214")

        assert rule_1.match(info)
        assert rule_2.match(info)
        assert rule_3.match(info)
        assert not rule_4.match(info)
        assert rule_5.match(info)
        assert not rule_6.match(info)
        assert not rule_7.match(info)
        
        


def test_cli(tmp_path):
    config_path = tmp_path / "config_path.jsonl"
    extractions_path = tmp_path / "extractions.jsonl"
    output_path = tmp_path / "output.csv"

    # Prepare config and save it on the disk
    config = [
        {
            "pattern": None,  # -> null
            "level": None,
            "source": None,
            "label": "accept",
        },
    ]

    JSONL.dump_jsonl(config, config_path)

    # Save extractions on the disk
    JSONL.dump_jsonl(EXTRACTED_TOPICS_EXAMPLE, extractions_path)

    args_and_opts = [
        extractions_path,
        config_path,
        output_path,
    ]

    # Run CLI
    topic_filter.run(*args_and_opts)

    # Assertions
    assert output_path.exists()

    output = pd.read_csv(output_path)

    assert output.columns.tolist() == [
        "path",
        "element_in_file",
        "accept",
        "source",
    ]

    assert output["accept"].sum() == len(EXTRACTED_TOPICS_EXAMPLE)

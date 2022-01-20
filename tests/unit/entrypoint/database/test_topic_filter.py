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

from bluesearch.entrypoint.database import topic_filter

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
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_bioarxiv_1",
        "source": "bioRxiv",
        "topics": {"article": {"Subject Area": "Neuroscience"}, "journal": {}},
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_bioarxiv_2",
        "source": "bioRxiv",
        "topics": {"article": {"Subject Area": "Microbiology"}, "journal": {}},
    },
    # medrxiv
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_medrxiv_1",
        "source": "medRxiv",
        "topics": {"article": {"Subject Area": "Epidemiology"}, "journal": {}},
    },
    {
        "metadata": {
            "bbs-version": "0.2.1.dev39",
            "created-date": "2022-01-18 10:06:12",
        },
        "path": "path_medrxiv_2",
        "source": "medRxiv",
        "topics": {
            "article": {"Subject Area": "Psychiatry and Clinical Psychology"},
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

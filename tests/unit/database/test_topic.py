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
import logging
import re
from unittest.mock import Mock

import pytest
import responses
from requests.exceptions import HTTPError

from bluesearch.database.topic import (
    extract_pubmed_id_from_pmc_file,
    get_topics_for_pmc_article,
)
from bluesearch.database.topic import (
    request_mesh_from_nlm_ta as request_mesh_from_nlm_ta_decorated,
)
from bluesearch.database.topic import (
    extract_info_from_zipfile, request_mesh_from_pubmed_id
)

# This function uses caching through @lru_cache. We want remove caching logic
# during tests.
request_mesh_from_nlm_ta = request_mesh_from_nlm_ta_decorated.__wrapped__


class TestGetMeshFromNlmTa:
    @pytest.mark.network
    def test_real_request_works(self):
        nlm_ta = "Trauma Surg Acute Care Open"
        expected_descriptors = {
            "Critical Care",
            "Emergency Treatment",
            "Wounds and Injuries",
        }

        mesh = request_mesh_from_nlm_ta(nlm_ta)

        assert mesh is not None
        assert len(mesh) == 3
        assert {item["descriptor"][0]["name"] for item in mesh} == expected_descriptors

    @responses.activate
    def test_normal_behaviour_works(self, test_data_path):
        with open(test_data_path / "nlmcatalog_response.txt") as f:
            body = f.read()

        responses.add(
            responses.GET,
            (
                "https://www.ncbi.nlm.nih.gov/nlmcatalog?"
                'term="Trauma Surg And Acute Care Open"[ta]&report=xml&format=text'
            ),
            body=body,
        )

        expected_output = [
            {
                "descriptor": [
                    {"name": "Critical Care", "major_topic": False, "ID": "D003422"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "name": "Emergency Treatment",
                        "major_topic": False,
                        "ID": "D004638",
                    }
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "name": "Wounds and Injuries",
                        "major_topic": False,
                        "ID": "D014947Q000517",
                    }
                ],
                "qualifiers": [
                    {"name": "prevention & control", "major_topic": False},
                    {"name": "surgery", "major_topic": True},
                ],
            },
        ]

        mesh = request_mesh_from_nlm_ta("Trauma Surg And Acute Care Open")
        assert mesh == expected_output

    def test_ampersands_are_flagged(self, caplog):
        nlm_ta = "Title with &#x0201c;ampersands&#x0201d"
        with caplog.at_level(logging.ERROR):
            meshes = request_mesh_from_nlm_ta(nlm_ta)
        assert meshes is None
        assert "Ampersands not allowed" in caplog.text

    @responses.activate
    def test_unexpected_response_doc_header_flagged(self, caplog):
        responses.add(
            responses.GET,
            re.compile(""),
            body="should start with a fixed header",
        )
        with caplog.at_level(logging.ERROR):
            meshes = request_mesh_from_nlm_ta("Some title")
        assert meshes is None
        assert "Unexpected response" in caplog.text

    @responses.activate
    def test_unexpected_response_doc_footer_flagged(self, caplog):
        header = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
            '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            "<pre>"
        )
        responses.add(
            responses.GET,
            re.compile(""),
            body=f"{header}the footer is missing though",
        )
        with caplog.at_level(logging.ERROR):
            meshes = request_mesh_from_nlm_ta("Some title")
        assert meshes is None
        assert "Unexpected response" in caplog.text

    @responses.activate
    def test_no_nlm_ta_found_gives_none(self, caplog):
        header = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
            '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            "<pre>"
        )
        footer = "</pre>"
        # Empty body means no results were found
        body = ""

        responses.add(
            responses.GET,
            re.compile(""),
            body=f"{header}{body}{footer}",
        )
        with caplog.at_level(logging.ERROR):
            meshes = request_mesh_from_nlm_ta("Some title")
        assert meshes is None
        assert "Empty body" in caplog.text


@responses.activate
def test_get_mesh_from_pubmedid(test_data_path):

    with open(test_data_path / "efetchpubmed_response.txt") as f:
        body = f.read()

    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        "db=pubmed&id=26633866,31755206&retmode=xml",
        body=body.encode("utf-8"),
    )

    expected_output = {
        "26633866": [
            {
                "descriptor": [
                    {"ID": "D001943", "major_topic": False, "name": "Breast Neoplasms"}
                ],
                "qualifiers": [
                    {"ID": "Q000191", "major_topic": True, "name": "economics"}
                ],
            },
            {
                "descriptor": [
                    {"ID": "D005060", "major_topic": False, "name": "Europe"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {"ID": "D005260", "major_topic": False, "name": "Female"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {"ID": "D017721", "major_topic": False, "name": "Hospital Costs"}
                ],
                "qualifiers": [
                    {
                        "ID": "Q000706",
                        "major_topic": False,
                        "name": "statistics & numerical data",
                    }
                ],
            },
            {
                "descriptor": [
                    {"ID": "D006801", "major_topic": False, "name": "Humans"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "ID": "D017059",
                        "major_topic": False,
                        "name": "Models, Econometric",
                    }
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "ID": "D012044",
                        "major_topic": False,
                        "name": "Regression Analysis",
                    }
                ],
                "qualifiers": [],
            },
        ],
        "31755206": [
            {
                "descriptor": [
                    {"ID": "D000328", "major_topic": False, "name": "Adult"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {"ID": "D005260", "major_topic": False, "name": "Female"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "ID": "D005377",
                        "major_topic": False,
                        "name": "Financial Management, Hospital",
                    }
                ],
                "qualifiers": [
                    {"ID": "Q000191", "major_topic": True, "name": "economics"}
                ],
            },
            {
                "descriptor": [
                    {
                        "ID": "D005380",
                        "major_topic": False,
                        "name": "Financing, Government",
                    }
                ],
                "qualifiers": [
                    {"ID": "Q000191", "major_topic": True, "name": "economics"}
                ],
            },
            {
                "descriptor": [
                    {"ID": "D005858", "major_topic": False, "name": "Germany"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "ID": "D006301",
                        "major_topic": False,
                        "name": "Health Services Needs and Demand",
                    }
                ],
                "qualifiers": [
                    {"ID": "Q000191", "major_topic": True, "name": "economics"}
                ],
            },
            {
                "descriptor": [
                    {"ID": "D006761", "major_topic": False, "name": "Hospitals"}
                ],
                "qualifiers": [
                    {"ID": "Q000639", "major_topic": True, "name": "trends"}
                ],
            },
            {
                "descriptor": [
                    {"ID": "D006801", "major_topic": False, "name": "Humans"}
                ],
                "qualifiers": [],
            },
            {
                "descriptor": [
                    {
                        "ID": "D007349",
                        "major_topic": False,
                        "name": "Insurance, Health, Reimbursement",
                    }
                ],
                "qualifiers": [
                    {"ID": "Q000191", "major_topic": True, "name": "economics"}
                ],
            },
            {
                "descriptor": [{"ID": "D008297", "major_topic": False, "name": "Male"}],
                "qualifiers": [],
            },
        ],
    }

    meshs = request_mesh_from_pubmed_id(["26633866", "31755206"])
    assert isinstance(meshs, dict)
    assert list(meshs.keys()) == ["26633866", "31755206"]
    assert meshs == expected_output

    responses.add(
        responses.GET,
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        "db=pubmed&id=0&retmode=xml",
        status=404,
    )

    with pytest.raises(HTTPError):
        request_mesh_from_pubmed_id(["0"])


def test_get_pubmedid(test_data_path):
    path = test_data_path / "jats_article.xml"
    pubmed_id = extract_pubmed_id_from_pmc_file(path)
    assert pubmed_id == "PMID"


def test_get_topics_for_pmc_article(test_data_path, monkeypatch):
    path = test_data_path / "jats_article.xml"
    fake_meshes = [
        {
            "descriptor": [
                {
                    "ID": "D017059",
                    "major_topic": False,
                    "name": "Models, Econometric",
                }
            ],
            "qualifiers": [],
        },
        {
            "descriptor": [
                {
                    "ID": "D012044",
                    "major_topic": False,
                    "name": "Regression Analysis",
                }
            ],
            "qualifiers": [],
        },
    ]
    request_mock = Mock(return_value=fake_meshes)
    monkeypatch.setattr(
        "bluesearch.database.topic.request_mesh_from_nlm_ta", request_mock
    )

    expected_output = ["Models, Econometric", "Regression Analysis"]
    journal_topics = get_topics_for_pmc_article(path)
    assert journal_topics == expected_output
    request_mock.assert_called_once()
    request_mock.assert_called_with("Journal NLM TA")


def test_extract_info_from_zipfile(test_data_path, tmp_path):
    test_xml_path = test_data_path / "jats_article.xml"

    # Put it inside of a `content` folder, zip it and then save in tmp_path


    zip_path = ...
    topic, journal = extract_info_from_zipfile(zip_path)


    assert topic == "Subject 1"
    assert journal == "bioRxiv"

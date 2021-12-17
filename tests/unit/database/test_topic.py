import re

import pytest
import responses
from defusedxml.ElementTree import ParseError
from requests.exceptions import HTTPError

from bluesearch.database.topic import (
    extract_pubmed_id_from_pmc_file,
    request_mesh_from_journal_title,
    request_mesh_from_pubmed_id,
)


class TestGetMeshFromNlmTa:
    @pytest.mark.network
    def test_real_request_works(self):
        nlm_ta = "Trauma Surgery And Acute Care Open"
        expected_descriptors = {
            "Critical Care",
            "Emergency Treatment",
            "Wounds and Injuries",
        }

        mesh = request_mesh_from_journal_title(nlm_ta)

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
                "term=Trauma Surgery And Acute Care Open[Title]&report=xml&format=text"
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

        mesh = request_mesh_from_journal_title("Trauma Surgery And Acute Care Open")
        assert mesh == expected_output

    def test_ampersands_are_flagged(self):
        nlm_ta = "Title with &#x0201c;ampersands&#x0201d"
        with pytest.raises(ValueError, match="Ampersands not allowed"):
            request_mesh_from_journal_title(nlm_ta)

    @responses.activate
    def test_unexpected_response_doc_header_flagged(self):
        responses.add(
            responses.GET,
            re.compile(""),
            body="should start with a fixed header",
        )
        with pytest.raises(RuntimeError, match="Unexpected response"):
            request_mesh_from_journal_title("Some title")

    @responses.activate
    def test_unexpected_response_doc_footer_flagged(self):
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
        with pytest.raises(RuntimeError, match="Unexpected response"):
            request_mesh_from_journal_title("Some title")

    @responses.activate
    def test_no_nlm_ta_found_gives_none(self):
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
        mesh = request_mesh_from_journal_title("Some title")
        assert mesh is None

    @responses.activate
    def test_invalid_xml_raises_correctly(self):
        header = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
            '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            "<pre>"
        )
        footer = "</pre>"
        body = "<<invalid-xml>"

        responses.add(
            responses.GET,
            re.compile(""),
            body=f"{header}{body}{footer}",
        )
        with pytest.raises(ParseError, match="The parsing did not work"):
            request_mesh_from_journal_title("Some title")

    @responses.activate
    def test_root_tag_is_checked(self):
        header = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
            '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            "<pre>"
        )
        footer = "</pre>"
        body = "<wrong-root-tag>Should be NCBICatalogRecord.</wrong-root-tag>"

        responses.add(
            responses.GET,
            re.compile(""),
            body=f"{header}{body}{footer}",
        )
        with pytest.raises(
            RuntimeError, match="Expected to find the NCBICatalogRecord tag"
        ):
            request_mesh_from_journal_title("Some title")


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



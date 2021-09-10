import pytest
import responses
from requests.exceptions import HTTPError

from bluesearch.database.topic import (
    extract_pubmed_id_from_pmc_file,
    request_mesh_from_nlm_ta,
    request_mesh_from_pubmed_id,
)


@responses.activate
def test_get_mesh_from_nlm_ta(test_data_path):
    with open(test_data_path / "nlmcatalog_response.txt") as f:
        body = f.read()

    responses.add(
        responses.GET,
        (
            "https://www.ncbi.nlm.nih.gov/nlmcatalog?"
            "term=Trauma Surg Acute Care Open[ta]&report=xml&format=text"
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
                {"name": "Emergency Treatment", "major_topic": False, "ID": "D004638"}
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

    mesh = request_mesh_from_nlm_ta("Trauma Surg Acute Care Open")
    assert mesh == expected_output


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
    path = test_data_path / "sample_file.xml"
    pubmed_id = extract_pubmed_id_from_pmc_file(path)
    assert pubmed_id == "PMID"

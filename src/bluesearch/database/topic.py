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
"""Utils for journal/articles topics."""
import pathlib
from typing import Dict, Iterable, List, Union
from xml.etree.ElementTree import Element  # nosec

import requests
from defusedxml import ElementTree


# Journal Topic
def get_mesh_from_nlm_ta(nlm_ta: str) -> List[Dict[str, Union[str, List[str]]]]:
    """Retrieve Medical Subject Heading from Journal's NLM Title Abbreviation.

    Parameters
    ----------
    nlm_ta : str
        NLM Title Abbreviation of Journal.

    Returns
    -------
    mesh :
    """
    nlm_ta_api = "+".join(nlm_ta.split(" "))
    url = (
        "https://www.ncbi.nlm.nih.gov/nlmcatalog/?"
        f"term={nlm_ta_api}%5Bta%5D&report=xml&format=xml"
    )

    response = requests.get(url)
    if not response.ok:
        raise ValueError("The request did not work")

    # The response is a fake xml,
    # we need to change some characters of the response to have a valid xml.
    text = (
        response.content.decode()
        .replace("<pre>", "")
        .replace("</pre>", "")
        .replace("\n", "")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("><", ">    <")
    )

    try:
        content = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        # Occurs when the number of results of the research is bigger than one.
        # It is the case for less than 1 % of the journal from PMC
        raise ElementTree.ParseError("The parsing did not work")

    mesh_headings = content.findall("./NLMCatalogRecord/MeshHeadingList/MeshHeading")
    meshs = get_mesh_from_nlmcatalog(mesh_headings)

    return meshs


# Article Topic
def get_mesh_from_pubmedid(pubmed_ids: Iterable[str]) -> Dict:
    """Retrieve Medical Subject Headings from Pubmed ID.

    Parameters
    ----------
    pubmed_ids : iterable of str
        List of Pubmed IDs.

    Returns
    -------
    pubmed_to_meshs : dict
        Dictionary containing Pubmed IDs as keys with corresponding
        Medical Subject Headings list as values.
    """
    pubmed_str = ",".join(pubmed_ids)
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={pubmed_str}&retmode=xml"
    )
    response = requests.get(url)

    if not response.ok:
        raise ValueError("The request did not work")

    content = ElementTree.fromstring(response.content.decode())
    pubmed_articles = content.findall("./PubmedArticle")
    pubmed_to_meshs = {}

    for article in pubmed_articles:
        pubmed_id = article.findall(
            "./PubmedData/ArticleIdList/ArticleId[@IdType='pubmed']"
        )[0].text
        mesh_headings = article.findall("./MedlineCitation/MeshHeadingList")
        meshs = get_mesh_from_pubmed(mesh_headings)
        pubmed_to_meshs[pubmed_id] = meshs

    return pubmed_to_meshs


# Utils
def get_pubmed_id(path: Union[str, pathlib.Path]) -> str:
    """Retrieve Pubmed ID from PMC XML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to PMC XML.

    Returns
    -------
    pubmed_id : str
        Pubmed ID of the given article
    """
    content = ElementTree.parse(path)
    article_id = content.findall("./front/article-meta/article-id[@pub-id-type='pmid']")

    if len(article_id) == 1:
        pubmed_id = article_id[0].text
    else:
        pubmed_id = None

    return pubmed_id


def get_mesh_from_nlmcatalog(mesh_headings: Element) -> List[Dict]:
    """Retrieve Medical Subject Headings from nlmcatalog parsing.

    Parameters
    ----------
    mesh_headings : Element
        XML parsing element containing all Medical Subject Headings.

    Returns
    -------
    mesh : list of dict
        List of dictionary containing Medical Subject Headings information.
    """
    meshs = []
    for mesh in mesh_headings:

        mesh_id = None
        if "URI" in mesh.attrib.keys():
            mesh_id = mesh.attrib["URI"].rpartition("/")[-1]

        descriptor_name = []
        qualifier_name = []

        for elem in mesh:
            major_topic = True if elem.attrib["MajorTopicYN"] == "Y" else False

            name = elem.text
            if name is not None:
                name = name.replace("&amp;", "&")

            if elem.tag == "DescriptorName":
                descriptor_name.append(
                    {"name": name, "major_topic": major_topic, "ID": mesh_id}
                )
            else:
                qualifier_name.append({"name": name, "major_topic": major_topic})

        meshs.append({"descriptor": descriptor_name, "qualifiers": qualifier_name})

    return meshs


def get_mesh_from_pubmed(mesh_headings: Element) -> List[Dict]:
    """Retrieve Medical Subject Headings from efetch pubmed parsing.

    Parameters
    ----------
    mesh_headings : Element
        XML parsing element containing all Medical Subject Headings.

    Returns
    -------
    mesh : list of dict
        List of dictionary containing Medical Subject Headings information.
    """
    meshs = []

    for mesh_heading in mesh_headings:

        for mesh in list(mesh_heading):

            descriptor_name = []
            qualifiers_name = []

            for info in list(mesh):

                attributes = info.attrib

                mesh_id = None
                if "UI" in attributes.keys():
                    mesh_id = attributes["UI"]

                major_topic = None
                if "MajorTopicYN" in attributes.keys():
                    major_topic = True if attributes["MajorTopicYN"] == "Y" else False

                if info.tag == "DescriptorName":
                    descriptor_name.append(
                        {"ID": mesh_id, "major_topic": major_topic, "name": info.text}
                    )
                else:
                    qualifiers_name.append(
                        {"ID": mesh_id, "major_topic": major_topic, "name": info.text}
                    )

            meshs.append({"descriptor": descriptor_name, "qualifiers": qualifiers_name})

    return meshs

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
"""Perform Name Entity Recognition (NER) on paragraphs."""
from __future__ import annotations

import logging
import os
from typing import Any

import elasticsearch
import numpy as np
import requests
import tqdm
from dotenv import load_dotenv
from elasticsearch.helpers import scan
import pandas as pd
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)


def run(
    client: elasticsearch.Elasticsearch,
    index: str = "paragraphs",
    ner_method: str = "ml",
    force: bool = False,
) -> None:
    """Run the NER pipeline on the paragraphs in the database.

    Parameters
    ----------
    client
        Elasticsearch client.
    index
        Name of the ES index.
    ner_method
        Method to use to perform NER.
    force
        If True, force the NER to be performed even in all paragraphs.
    """
    # get NER method function and url
    if ner_method == "ml":
        url = os.environ["BENTOML_NER_ML_URL"]
    elif ner_method == "ruler":
        url = os.environ["BENTOML_NER_RULER_URL"]
    else:
        raise ValueError("The ner_method should be either 'ml' or 'ruler'.")

    # get paragraphs without NER unless force is True
    if force:
        query: dict[str, Any] = {"match_all": {}}
    else:
        query = {
            "bool": {"must_not": {"exists": {"field": f"ner_{ner_method}_json_v2"}}}
        }
    paragraph_count = client.count(index=index, query=query)["count"]
    logger.info(
        f"There are {paragraph_count} paragraphs without NER {ner_method} results."
    )

    # performs NER for all the documents
    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Updating NER",
    )
    for hit in scan(client, query={"query": query}, index=index, scroll="12h"):
        try:
            results = run_ner_model_remote(
                hit["_source"]["text"],
                url,
                ner_method,
            )
            client.update(
                index=index, doc={f"ner_{ner_method}_json_v2": results}, id=hit["_id"]
            )

            progress.update(1)
            logger.info(
                f"Updated NER for paragraph {hit['_id']}, progress: {progress.n}"
            )
        except Exception as e:
            print(e)
            logger.error(f"Error in paragraph {hit['_id']}, progress: {progress.n}")

    progress.close()


def run_ner_model_remote(text: str, url: str, source: str) -> list[dict]:
    """Run NER on the remote server for a specific paragraph text.

    Parameters
    ----------
    text
        Text to perform NER on.
    url
        URL of the remote server.
    source
        Source model of the NER results.

    Returns
    -------
    results
        List of dictionaries with the NER results.
    """
    url = "http://" + url + "/predict"

    response = requests.post(
        url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        data=text.encode("utf-8"),
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    results = response.json()

    out = []
    for res in results:
        row = {}
        row["entity_type"] = res["entity_group"]
        row["entity"] = res["word"]
        row["start"] = res["start"]
        row["end"] = res["end"]
        row["score"] = 0 if source == "ruler" else res["score"]
        row["source"] = source
        out.append(row)

    return out


def handle_conflicts(results_paragraph: list[dict]) -> list[dict]:
    """Handle conflicts between the NER pipeline and the entity ruler."""
    # if there is only one entity, it will be kept
    if len(results_paragraph) <= 1:
        return results_paragraph

    temp = sorted(
        results_paragraph,
        key=lambda x: (-(x["end"] - x["start"]), x["source"]),
    )

    results_cleaned: list[dict] = []

    array = np.zeros(max([x["end"] for x in temp]))
    for res in temp:
        add_one = 1 if res["word"][0] == " " else 0
        sub_one = 1 if res["word"][-1] == " " else 0
        if len(results_cleaned) == 0:
            results_cleaned.append(res)
            array[res["start"] + add_one : res["end"] - sub_one] = 1
        else:
            if array[res["start"] + add_one : res["end"] - sub_one].sum() == 0:
                results_cleaned.append(res)
                array[res["start"] + add_one : res["end"] - sub_one] = 1

    results_cleaned.sort(key=lambda x: x["start"])
    return results_cleaned


def retrieve_csv(
    client: elasticsearch.Elasticsearch,
    index: str = "paragraphs",
    ner_method: str = "both",
    output_path: str = "./",
) -> None:
    """Retrieve the NER results from the database and save them in a csv file.

    Parameters
    ----------
    client
        Elasticsearch client.
    index
        Name of the ES index.
    ner_method
        Method to use to perform NER.
    """
    now = datetime.now().strftime('%d_%m_%Y_%H_%M')

    if ner_method == "both":
        query = {
            "bool": {
                "filter": [
                    {"exists": {"field": "ner_ml_json_v2"}},
                    {"exists": {"field": "ner_ruler_json_v2"}},
                ]
            }
        }
    elif ner_method in ["ml", "ruler"]:
        query = {"exists": {"field": f"ner_{ner_method}_json_v2"}}
    else:
        raise ValueError("The ner_method should be either 'ml', 'ruler' or 'both'.")

    paragraph_count = client.count(index=index, query=query)["count"]
    logger.info(
        f"There are {paragraph_count} paragraphs with NER {ner_method} results."
    )

    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Retrieving NER",
    )
    results = []
    for hit in scan(client, query={"query": query}, index=index, scroll="12h"):
        if ner_method == "both":
            results_paragraph = [
                *hit["_source"]["ner_ml_json_v2"],
                *hit["_source"]["ner_ruler_json_v2"],
            ]
            results_paragraph = handle_conflicts(results_paragraph)
        else:
            results_paragraph = hit["_source"][f"ner_{ner_method}_json_v2"]

        for res in results_paragraph:
            row = {}
            row["entity_type"] = res["entity_type"]
            row["entity"] = res["entity"]
            row["start"] = res["start"]
            row["end"] = res["end"]
            row["source"] = res["source"]
            row["paragraph_id"] = hit["_id"]
            row["article_id"] = hit["_source"]["article_id"]
            results.append(row)

        progress.update(1)
        logger.info(
            f"Retrieved NER for paragraph {hit['_id']}, progress: {progress.n}"
        )

    progress.close()

    df = pd.DataFrame(results)
    df.to_csv(f"{output_path}/ner_es_results{ner_method}_{now}.csv", index=False)

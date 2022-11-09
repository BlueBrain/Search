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
"""Perform Relation Extraction (RE) using a remote server."""
from __future__ import annotations

import json
import logging
import os
import time
from itertools import product
from multiprocessing import Pool
from typing import Any

import elasticsearch
import requests
import tqdm
from dotenv import load_dotenv
from elasticsearch.helpers import scan

from bluesearch.k8s import connect
from bluesearch.k8s.ner import handle_conflicts

load_dotenv()

logger = logging.getLogger(__name__)


def run(
    client: elasticsearch.Elasticsearch,
    version: str,
    index: str = "paragraphs",
    force: bool = False,
    n_threads: int = 4,
    run_async: bool = True,
) -> None:
    """Run the RE pipeline on the paragraphs in the database.

    Parameters
    ----------
    client
        Elasticsearch client.
    version
        Version of the RE pipeline.
    index
        Name of the ES index.
    force
        If True, force the RE to be performed even in all paragraphs.
    n_threads
        Number of threads to use.
    run_async
        If True, run the RE asynchronously.
    """
    url = os.environ["BENTOML_RE_ML_URL"]

    # get paragraphs without NER unless force is True
    if force:
        query: dict[str, Any] = {"match_all": {}}
    else:
        query = {"bool": {"must_not": {"term": {"re_version": version}}}}
    paragraph_count = client.options(request_timeout=30).count(
        index=index, query=query
    )["count"]
    logger.info(f"There are {paragraph_count} paragraphs without RE results.")

    # performs NER for all the documents
    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Updating RE",
    )
    if run_async:
        # start a pool of workers
        pool = Pool(processes=n_threads)
        open_threads = []
        for hit in scan(client, query={"query": query}, index=index, scroll="24h"):
            # add a new thread to the pool
            res = pool.apply_async(
                run_re_model_remote,
                args=(
                    hit,
                    url,
                    index,
                    version,
                ),
            )
            open_threads.append(res)
            # check if any thread is done
            open_threads = [thr for thr in open_threads if not thr.ready()]
            # wait if too many threads are running
            while len(open_threads) > n_threads:
                time.sleep(0.1)
                open_threads = [thr for thr in open_threads if not thr.ready()]
            progress.update(1)
        # wait for all threads to finish
        pool.close()
        pool.join()
    else:
        for hit in scan(client, query={"query": query}, index=index, scroll="24h"):
            run_re_model_remote(
                hit=hit,
                url=url,
                index=index,
                version=version,
                client=client,
            )
            progress.update(1)

    progress.close()


def prepare_text_for_re(
    text: str,
    subj: dict,
    obj: dict,
    subject_symbols: tuple[str, str] = ("[[ ", " ]]"),
    object_symbols: tuple[str, str] = ("<< ", " >>"),
) -> str:
    """Add the subj and obj annotation to the text."""
    if subj["start"] < obj["start"]:
        first, second = subj, obj
        first_symbols, second_symbols = subject_symbols, object_symbols
    else:
        first, second = obj, subj
        first_symbols, second_symbols = object_symbols, subject_symbols

    attribute = "entity"

    part_1 = text[: first["start"]]
    part_2 = f"{first_symbols[0]}{first[attribute]}{first_symbols[1]}"
    part_3 = text[first["end"] : second["start"]]
    part_4 = f"{second_symbols[0]}{second[attribute]}{second_symbols[1]}"
    part_5 = text[second["end"] :]

    out = part_1 + part_2 + part_3 + part_4 + part_5

    return out


def run_re_model_remote(
    hit: dict[str, Any],
    url: str,
    index: str | None = None,
    version: str | None = None,
    client: elasticsearch.Elasticsearch | None = None,
) -> list[dict[str, Any]] | None:
    """Perform RE on a paragraph using a remote server.

    Parameters
    ----------
    hit
        Elasticsearch hit.
    url
        URL of the Relation Extraction (RE) server.
    index
        Name of the ES index.
    version
        Version of the Relation Extraction pipeline.
    """
    if client is None and index is None and version is None:
        logger.info("Running RE in inference mode only.")
    elif client is None and index is not None and version is not None:
        client = connect.connect()
    elif client is None and (index is not None or version is not None):
        raise ValueError("Index and version should be both None or not None.")

    url = "http://" + url + "/predict"

    matrix: list[tuple[str, str]] = [
        ("BRAIN_REGION", "ORGANISM"),
        ("CELL_COMPARTMENT", "CELL_TYPE"),
        ("CELL_TYPE", "BRAIN_REGION"),
        ("CELL_TYPE", "ORGANISM"),
        ("GENE", "BRAIN_REGION"),
        ("GENE", "CELL_COMPARTMENT"),
        ("GENE", "CELL_TYPE"),
        ("GENE", "ORGANISM"),
    ]

    text = hit["_source"]["text"]
    ner_ml = hit["_source"]["ner_ml_json_v2"]
    ner_ruler = hit["_source"]["ner_ruler_json_v2"]

    results_cleaned = handle_conflicts([*ner_ml, *ner_ruler])

    texts = []
    sub_obj = []
    for subj, obj in product(results_cleaned, results_cleaned):
        if subj == obj:
            continue
        if (subj["entity_type"], obj["entity_type"]) in matrix:
            text_processed = prepare_text_for_re(text, subj, obj)
            texts.append(text_processed)
            sub_obj.append((subj, obj))

    response = requests.post(
        url,
        headers={"accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(texts).encode("utf-8"),
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    result = response.json()
    out = []
    if result:
        for (subj, obj), res in zip(sub_obj, result):
            row = {}
            row["label"] = res["label"]
            row["score"] = res["score"]
            row["subject_entity_type"] = subj["entity_type"]
            row["subject_entity"] = subj["entity"]
            row["subject_start"] = subj["start"]
            row["subject_end"] = subj["end"]
            row["subject_source"] = subj["source"]
            row["object_entity_type"] = obj["entity_type"]
            row["object_entity"] = obj["entity"]
            row["object_start"] = obj["start"]
            row["object_end"] = obj["end"]
            row["object_source"] = obj["source"]
            row["source"] = "ml"
            out.append(row)

    if client is not None and index is not None and version is not None:
        # update the RE field in the document
        client.update(index=index, doc={"re": out}, id=hit["_id"])
        # update the version of the RE
        client.update(index=index, doc={"re_version": version}, id=hit["_id"])
        return None
    else:
        return out


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)
    client = connect.connect()
    run(client, version="v1", run_async=False, n_threads=8)

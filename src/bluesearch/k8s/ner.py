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
"""Perform Name Entity Recognition (NER) on a paragraph."""
import logging
import os
from typing import Any

import elasticsearch
import numpy as np
import requests
import tqdm
from dotenv import load_dotenv
from elasticsearch.helpers import scan

load_dotenv()

logger = logging.getLogger(__name__)


def run(
    client: elasticsearch.Elasticsearch,
    index: str = "paragraphs",
    ner_method: str = "both",
    force: bool = False,
) -> None:
    """Runs the NER pipeline on the paragraphs in the database.

    Parameters
    ----------
    client
        Elasticsearch client.
    index
        Name of the ES index.
    ner_method
        Method to use to perform NER.
    """

    # get paragraphs without NER unless force is True
    if force:
        query: dict[str, Any] = {"match_all": {}}
    else:
        query = {"bool": {"must_not": {"exists": {"field": "ner"}}}}
    paragraph_count = client.count(index=index, query=query)["count"]
    logger.info("There are {paragraph_count} paragraphs without embeddings")

    # creates NER for all the documents
    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Updating NER",
    )
    for hit in scan(client, query={"query": query}, index=index):
        if ner_method == "both":
            results_ml = run_ml_ner(
                hit["_source"]["text"],
                os.environ["BENTOML_NER_ML_URL"]
            )
            results_ruller = run_ruler_ner(
                hit["_source"]["text"],
                os.environ["BENTOML_NER_RULER_URL"]
            )

            client.update(index=index, doc={"ner_ml": results_ml}, id=hit["_id"])
            client.update(index=index, doc={"ner_ruler": results_ruller}, id=hit["_id"])

        elif ner_method == "ml":
            results = run_ml_ner(
                hit["_source"]["text"],
                os.environ["BENTOML_NER_RULER_URL"]
            )
            client.update(index=index, doc={"ner_ml": results}, id=hit["_id"])
        elif ner_method == "ruler":
            results = run_ruler_ner(
                hit["_source"]["text"],
                os.environ["BENTOML_NER_RULER_URL"]
            )
            client.update(index=index, doc={"ner_ruler": results}, id=hit["_id"])
        else:
            raise ValueError(f"Unknown NER method: {ner_method}")

        progress.update(1)
        logger.info(f"Updated NER for paragraph {hit['_id']}, progress: {progress.n}")

    progress.close()


def run_ml_ner(text: str, url: str) -> list[dict]:
    """Runs the NER pipeline on the paragraphs in the database.

    Parameters
    ----------
    text
        Text to perform NER on.
    article_id
        Id of the article.
    paragraph_id
        Id of the paragraph.
    ml_model
        Name of the ML model to use.
    """
    url = "http://" + url + "/predict"

    response = requests.post(
        url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        data=text,
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    results = response.json()

    out = []
    for res in results:
        row = {}
        row["entity"] = res["entity_group"]
        row["word"] = res["word"]
        row["start"] = res["start"]
        row["end"] = res["end"]
        row["source"] = "ML"
        out.append(row)

    return out


def run_ruler_ner(
    text: str, url: str
) -> list[dict]:
    """Runs the NER pipeline on the paragraphs in the database.

    Parameters
    ----------
    text
        Text to perform NER on.
    article_id
        Id of the article.
    paragraph_id
        Id of the paragraph.
    ruler_model
        Name of the entity ruler model to use.
    """
    url = "http://" + url + "/predict"

    response = requests.post(
        url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        data=text,
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    results = response.json()

    out = []
    for res in results:
        row = {}
        row["entity"] = res["entity"]
        row["word"] = res["word"]
        row["start"] = res["start"]
        row["end"] = res["end"]
        row["source"] = "RULES"
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

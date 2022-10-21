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
"""Embed the paragraphs in the database."""
from __future__ import annotations

import functools
import logging
import os
from typing import Any

import elasticsearch
import numpy as np
import requests
import tqdm
from dotenv import load_dotenv
from elasticsearch.helpers import scan

from bluesearch.embedding_models import SentTransformer

load_dotenv()

logger = logging.getLogger(__name__)


def embed(
    client: elasticsearch.Elasticsearch,
    index: str = "paragraphs",
    embedding_method: str = "seldon",
    model_name: str = "minilm",
    namespace: str = "seldon",
    polling: str = "mean",
    force: bool = False,
) -> None:
    """Embed the paragraphs in the database locally.

    Parameters
    ----------
    client
        Elasticsearch client.
    index
        Name of the ES index.
    embedding_method
        Method to use to embed the paragraphs.
    model_name
        Name of the model to use for the embedding.
    namespace
        Namespace of the Seldon deployment.
    polling
        Polling method to use for the Seldon deployment.
    """
    if embedding_method == "seldon":
        embed = functools.partial(
            embed_seldon, namespace=namespace, model_name=model_name, polling=polling
        )
    elif embedding_method == "bentoml":
        embed = functools.partial(embed_bentoml, model_name=model_name)
    elif embedding_method == "local":
        embed = functools.partial(
            embed_locally,
            model_name=model_name,
        )
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")

    # get paragraphs without embeddings
    if force:
        query: dict[str, Any] = {"query": {"match_all": {}}}
    else:
        query: dict[str, Any] = {
            "query": {"bool": {"must_not": {"exists": {"field": "embedding"}}}}
        }
    paragraph_count = client.count(index=index, query=query)["count"]
    logger.info("There are {paragraph_count} paragraphs without embeddings")

    # creates embeddings for all the documents withouts embeddings and updates them
    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Updating embeddings",
    )
    for hit in scan(client, query={"query": query}, index=index):
        emb = embed(hit["_source"]["text"])
        client.update(index=index, doc={"embedding": emb}, id=hit["_id"])
        progress.update(1)


def embed_locally(
    text: str, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
) -> list[float]:
    """Embed the paragraphs in the database locally.

    Parameters
    ----------
    text
        Text to embed.
    model_name
        Name of the model to use for the embedding.

    Returns
    -------
    embedding
        Embedding of the text.
    """
    model = SentTransformer(model_name)
    emb = model.embed(text)
    if not model.normalized:
        emb /= np.linalg.norm(emb)
    return emb.tolist()


def embed_seldon(
    text: str,
    namespace: str = "seldon",
    model_name: str = "minilm",
    polling: str = "mean",
) -> list[float]:
    """Embed the paragraphs in the database using Seldon.

    Parameters
    ----------
    text
        Text to embed.
    namespace
        Namespace of the Seldon deployment.
    model_name
        Name of the Seldon deployment.
    polling
        Polling method to use for the Seldon deployment.

    Returns
    -------
    embedding
        Embedding of the text.
    """
    url = (
        "http://"
        + os.environ["SELDON_URL"]
        + "/seldon/"
        + namespace
        + "/"
        + model_name
        + "/v2/models/transformer/infer"
    )

    # create payload
    response = requests.post(
        url,
        json={
            "inputs": [
                {
                    "name": "args",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": text,
                    "parameters": {"content_type": "str"},
                }
            ]
        },
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    # convert the response to a numpy array
    tensor = response.json()["outputs"][0]["data"][0]
    tensor = tensor[3:-3].split("], [")
    tensor = np.vstack([np.array(t.split(", "), dtype=np.float32) for t in tensor])

    # apply the polling method
    if polling:
        if polling == "max":
            tensor = np.max(tensor, axis=0)
        elif polling == "mean":
            tensor = np.mean(tensor, axis=0)

    # normalize the embedding
    tensor /= np.linalg.norm(tensor)

    return tensor.tolist()


def embed_bentoml(
    text: str, model_name: str = "minilm", polling: str = "mean"
) -> list[float]:
    """Embed the paragraphs in the database using BentoML.

    Parameters
    ----------
    text
        Text to embed.
    model_name
        Name of the BentoML deployment.

    Returns
    -------
    embedding
        Embedding of the text.
    """
    url = "http://" + os.environ["BENTOML_EMBEDDING_URL"] + "/" + model_name

    # create payload
    response = requests.post(
        url,
        headers={"accept": "application/json", "Content-Type": "text/plain"},
        data=text,
    )

    if not response.status_code == 200:
        raise ValueError("Error in the request")

    # convert the response to a numpy array
    tensor = response.json()
    tensor = np.vstack(tensor[0])

    # apply the polling method
    if polling:
        if polling == "max":
            tensor = np.max(tensor, axis=0)
        elif polling == "mean":
            tensor = np.mean(tensor, axis=0)

    # normalize the embedding
    tensor /= np.linalg.norm(tensor)

    return tensor

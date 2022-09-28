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

import logging

import elasticsearch
import numpy as np
import tqdm
from elasticsearch.helpers import scan

from bluesearch.embedding_models import SentTransformer

logger = logging.getLogger(__name__)


def embed_locally(
    client: elasticsearch.Elasticsearch,
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    index: str = "paragraphs",
) -> None:
    """Embed the paragraphs in the database locally.

    Parameters
    ----------
    client
        Elasticsearch client.
    model_name
        Name of the model to use for the embedding.
    """
    model = SentTransformer(model_name)

    # get paragraphs without embeddings
    query = {"bool": {"must_not": {"exists": {"field": "embedding"}}}}
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
        emb = model.embed(hit["_source"]["text"])
        if not model.normalized:
            emb /= np.linalg.norm(emb)
        client.update(index=index, doc={"embedding": emb.tolist()}, id=hit["_id"])
        progress.update(1)

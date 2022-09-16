import logging

import elasticsearch
import tqdm
from elasticsearch.helpers import scan

from bluesearch.embedding_models import SentTransformer

logger = logging.getLogger(__name__)


def embed_locally(
    client: elasticsearch.Elasticsearch,
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
) -> None:

    model = SentTransformer(model_name)

    # get paragraphs without embeddings
    query = {"query": {"bool": {"must_not": {"exists": {"field": "embedding"}}}}}
    paragraph_count = client.count(index="paragraphs", query=query)["count"]
    print(f"There are {paragraph_count} paragraphs without embeddings")

    # creates embeddings for all the documents without embeddings and updates them
    progress = tqdm.tqdm(
        total=paragraph_count,
        position=0,
        unit=" Paragraphs",
        desc="Updating embeddings",
    )
    for hit in scan(client, query=query, index="paragraphs"):
        emb = model.embed(hit["_source"]["text"])
        client.update(
            index="paragraphs", doc={"embedding": emb.tolist()}, id=hit["_id"]
        )
        progress.update(1)

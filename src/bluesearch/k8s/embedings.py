import logging

import elasticsearch
import tqdm
from elasticsearch.helpers import scan

from bluesearch.embedding_models import SentTransformer
from bluesearch.k8s.connect import connect

logger = logging.getLogger(__name__)


def embed_locally(
    client: elasticsearch.Elasticsearch = connect(),
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
) -> None:

    model = SentTransformer(model_name)

    # get paragraphs without embeddings
    query = {"query": {"bool": {"must_not": {"exists": {"field": "embedding"}}}}}
    paragraph_count = client.count(index="paragraphs", body=query)[  # type: ignore
        "count"
    ]
    logger.info("There are {paragraph_count} paragraphs without embeddings")

    # creates embeddings for all the documents withouts embeddings and updates them
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


if __name__ == "__main__":
    embed_locally()

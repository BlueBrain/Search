from typing import Optional

import numpy as np
import tensorflow_hub as hub
from cord19q.embeddings import Embeddings
from cord19q.tokenizer import Tokenizer


class CustomEmbeddings(Embeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(">>> CustomEmbeddings <<<")
        self.univ_sent_emb = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def transform(self, query: str, method: str, id: Optional[str] = None, tags=None):
        if method == "fasttext_bm25":
            document = Tokenizer.tokenize(query)
            weights = self.scoring.weights(document) if self.scoring else None
            vector = self.lookup(document[1])
            if weights and [x for x in weights if x > 0]:
                embedding = np.average(vector, weights=np.array(weights, dtype=np.float32), axis=0)
            else:
                embedding = np.mean(vector, axis=0)

        elif method == "univ_sent_emb":
            embedding = self.univ_sent_emb([query]).numpy().squeeze()

        else:
            raise Exception('method unknown')

        embedding = self.removePC(embedding) if self.lsa else embedding

        return self.normalize(embedding) if self.embeddings else embedding

    def search(self, query: str, method: str, limit: int = 3):
        embedding = self.transform(query, method)

        self.embeddings.nprobe = 6
        results = self.embeddings.search(embedding.reshape(1, -1), limit)

        return list(zip(results[1][0].tolist(), (results[0][0]).tolist()))

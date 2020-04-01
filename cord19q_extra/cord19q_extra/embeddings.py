import numpy as np
from cord19q.embeddings import Embeddings


class CustomEmbeddings(Embeddings):

    def transform(self, document):
        weights = self.scoring.weights(document) if self.scoring else None

        if weights and [x for x in weights if x > 0]:
            # TODO
            embedding = np.average(self.lookup(document[1]), weights=np.array(weights, dtype=np.float32), axis=0)
        else:
            # TODO
            embedding = np.mean(self.lookup(document[1]), axis=0)

        embedding = self.removePC(embedding) if self.lsa else embedding

        return self.normalize(embedding) if self.embeddings else embedding

    def search(self, tokens, limit=3):
        # TODO
        embedding = self.transform((None, tokens, None))

        self.embeddings.nprobe = 6
        results = self.embeddings.search(embedding.reshape(1, -1), limit)

        return list(zip(results[1][0].tolist(), (results[0][0]).tolist()))

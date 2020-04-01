from cord19q.query import Query
from cord19q.tokenizer import Tokenizer


class CustomQuery(Query):

    @staticmethod
    def search(embeddings, cur, query, topn):
        results = []

        # TODO
        query = Tokenizer.tokenize(query)

        for uid, score in embeddings.search(query, topn):
            cur.execute("SELECT Article, Text FROM sections WHERE id = ?", [uid])
            results.append((uid, score) + cur.fetchone())

        return results

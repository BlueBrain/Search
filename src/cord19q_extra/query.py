from cord19q.models import Models
from cord19q.query import Query

from .models import CustomModels


class CustomQuery(Query):

    @staticmethod
    def search(embeddings, cur, query, method: str, topn):
        results = []

        for uid, score in embeddings.search(query, method, topn):
            cur.execute("SELECT sentence_id, text FROM sentences WHERE sentence_id = ?", [uid])
            results.append((uid, score) + cur.fetchone())

        return results

    @staticmethod
    def query(embeddings, db, query, method: str, topn):
        # Default to 10 results if not specified
        topn = topn if topn else 10

        cur = db.cursor()

        print(Query.render("#Query: %s" % query, theme="729.8953") + "\n")

        # Query for best matches
        results = CustomQuery.search(embeddings, cur, query, method, topn)

        # Extract top sections as highlights
        print(Query.render("# Highlights"))
        for highlight in Query.highlights(results, int(topn / 5)):
            print(Query.render("## - %s" % Query.text(highlight)))

        print()

        # Get results grouped by document
        documents = Query.documents(results)

        print(Query.render("# Articles") + "\n")

        # Print each result, sorted by max score descending
        for uid in sorted(documents, key=lambda k: sum([x[0] for x in documents[k]]),
                          reverse=True):
            cur.execute(
                "SELECT title, authors, date, journal, article_id, url from articles where article_id = ?",
                [uid])
            article = cur.fetchone()

            print("Title: %s" % article[0])
            print("Authors: %s" % Query.authors(article[1]))
            print("Published: %s" % Query.date(article[2]))
            print("Publication: %s" % article[3])
            print("Id: %s" % article[4])
            print("Reference: %s" % article[5])

            # Print top matches
            for score, text in documents[uid]:
                print(Query.render("## - (%.4f): %s" % (score, text), html=False))

            print()

    @staticmethod
    def run(query, method: str, topn=None, path=None):
        # Load model
        embeddings, db = CustomModels.load(path)

        # Query the database
        CustomQuery.query(embeddings, db, query, method, topn)

        # Free resources
        Models.close(db)

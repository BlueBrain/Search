import sqlite3

from cord19q.index import Index
from cord19q.tokenizer import Tokenizer


class CustomIndex(Index):

    @staticmethod
    def stream(dbfile):
        db = sqlite3.connect(dbfile)
        cur = db.cursor()

        cur.execute("""SELECT sentence_id, text FROM sentences
                WHERE sha IN (SELECT sha FROM article_id_2_sha WHERE article_id IN 
                (SELECT article_id FROM articles WHERE has_covid19_tag is True))""")

        count = 0
        for row in cur:
            # TODO
            tokens = Tokenizer.tokenize(row[1])

            document = (row[0], tokens, None)

            count += 1
            if count % 1000 == 0:
                print("Streamed %d documents" % (count))

            if tokens:
                yield document

        print("Iterated over %d total rows" % (count))

        db.close()

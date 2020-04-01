import os
import sqlite3

from cord19q.models import Models

from .embeddings import CustomEmbeddings


class CustomModels(Models):

    @staticmethod
    def load(path):
        # Default path if not provided
        if not path:
            path = Models.modelPath()

        dbfile = os.path.join(path, "articles.sqlite")

        if os.path.isfile(os.path.join(path, "config")):
            print("Loading model from %s" % path)
            embeddings = CustomEmbeddings()
            embeddings.load(path)
        else:
            print("ERROR: loading model: ensure model is present")
            raise FileNotFoundError("Unable to load model from %s" % path)

        # Connect to database file
        db = sqlite3.connect(dbfile)

        return (embeddings, db)

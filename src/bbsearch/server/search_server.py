from flask import request, jsonify

from ..local_searcher import LocalSearcher


class SearchServer:

    def __init__(self,
                 app,
                 trained_models_path,
                 embeddings_path,
                 databases_path):
        self.app = app
        self.local_searcher = LocalSearcher(trained_models_path, embeddings_path, databases_path)

        app.route("/", methods=["POST"])(self.query)

    def query(self):
        if request.is_json:
            json_request = request.get_json()
            sentence_ids, similarities, stats = self.local_searcher.query(**json_request)
            result = dict(
                sentence_ids=sentence_ids,
                similarities=similarities,
                stats=stats)
        else:
            result = {}

        response = jsonify(result)
        return response

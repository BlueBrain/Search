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

            which_model = json_request.pop("which_model")
            k = json_request.pop("k")
            query_text = json_request.pop("query_text")

            sentence_ids, similarities, stats = self.local_searcher.query(
                which_model,
                k,
                query_text,
                **json_request)

            sentence_ids = [idx.item() for idx in sentence_ids],
            similarities = [sim.item() for sim in similarities],
            stats = stats
        else:
            sentence_ids = None
            similarities = None
            stats = None

        response = dict(
            sentence_ids=sentence_ids,
            similarities=similarities,
            stats=stats)
        response_json = jsonify(response)

        return response_json

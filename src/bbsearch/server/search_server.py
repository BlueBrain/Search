"""The search server."""

from flask import request, jsonify

from ..local_searcher import LocalSearcher


class SearchServer:
    """The BBS search server.

    Parameters
    ----------
    app : flask.Flask
        The Flask app wrapping the server.
    trained_models_path : str or pathlib.Path
        The folder containing pre-trained models.
    embeddings_path : str or pathlib.Path
        The folder containing pre-computed embeddings.
    databases_path : str or pathlib.Path
        The folder containing the SQL databases.
    """

    def __init__(self,
                 app,
                 trained_models_path,
                 embeddings_path,
                 databases_path):
        self.app = app
        self.local_searcher = LocalSearcher(trained_models_path, embeddings_path, databases_path)

        app.route("/", methods=["POST"])(self.query)

    def query(self):
        """The main query callback routed to "/".

        Returns
        -------
        response_json : flask.Response
            The JSON response to the query.
        """
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

            response = dict(
                sentence_ids=sentence_ids.tolist(),
                similarities=similarities.tolist(),
                stats=stats)
        else:
            response = dict(
                sentence_ids=None,
                similarities=None,
                stats=None)

        response_json = jsonify(response)

        return response_json

import logging

from flask import jsonify

logger = logging.getLogger(__name__)


class MiningServer:

    def __init__(self, app):
        self.version = "1.0"
        self.name = "MiningServer"

        self.app = app
        self.app.route("/", methods=["POST"])(self.pipeline)
        self.app.route("/identify", methods=["POST"])(self.identify)

    def identify(self):
        response = {
            "name": self.name,
            "version": self.version,
        }

        return jsonify(response)

    def pipeline(self):
        csv = ""
        return csv

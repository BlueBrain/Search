class InvalidUsage(Exception):
    """An exception used in the REST API server

    The code was largely copied from
    https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """

    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is None:
            self.status_code = 400
        else:
            self.status_code = status_code

    def to_dict(self):
        rv = dict()
        rv['message'] = self.message
        return rv

"""Custom exceptions."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


class InvalidUsage(Exception):
    """An exception used in the REST API server.

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
        """Generate a dictionary."""
        rv = {}
        rv["message"] = self.message
        return rv

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
"""connects to ES."""
import logging
import os

import urllib3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
urllib3.disable_warnings()

logger = logging.getLogger(__name__)


def connect() -> Elasticsearch:
    """Return a client connect ES."""
    client = Elasticsearch(
        os.environ["ES_URL"],
        basic_auth=("elastic", os.environ["ES_PASS"]),
        verify_certs=False,
    )

    if not client.ping():
        raise RuntimeError(f"Cannot connect to ES: {os.environ['ES_URL']}")

    logger.info("Connected to ES")

    return client


if __name__ == "__main__":
    connect()

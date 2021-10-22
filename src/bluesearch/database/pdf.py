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
"""Module for PDF conversion."""
import requests


def grobid_pdf_to_tei_xml(pdf_content: bytes, host: str, port: str) -> str:
    """Convert PDF file to TEI XML using GROBID server.

    This function uses the GROBID API service to convert PDF to a TEI XML format.
    In order to setup GROBID server, follow the instructions from
    https://grobid.readthedocs.io/en/latest/Grobid-docker/.

    Parameters
    ----------
    pdf_content
        PDF content
    host
        Host of the GROBID server.
    port
        Port of the GROBID server.

    Returns
    -------
    str
        TEI XML parsing of the PDF content.
    """
    url = f"http://{host}:{port}/api/processFulltextDocument"
    files = {"input": pdf_content}
    headers = {"Accept": "application/xml"}
    timeout = 60

    response = requests.post(
        url=url,
        files=files,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text

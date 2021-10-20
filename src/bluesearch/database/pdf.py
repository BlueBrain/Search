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


def convert_pdfs_to_tei_xml(pdf_content: bytes, host: str, port: str) -> str:
    """Convert PDF file to TEI XML.

    Parameters
    ----------
    pdf_content
        Pdf content
    host
        Host of the server.
    port
        Port of the server.

    Returns
    -------
    str
        TEI XML parsing of the pdf content.
    """
    url = f"http://{host}:{port}/api/processFulltextDocument"
    files = {
        "input": (
            "",
            pdf_content,
            "application/pdf",
            {"Expires": "0"},
        )
    }
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

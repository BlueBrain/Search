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
import asyncio
import logging

import aiohttp
import requests

logger = logging.getLogger(__name__)


def grobid_is_alive(host: str, port: int) -> bool:
    """Test if the GROBID server is alive.

    This server API is documented here:
    https://grobid.readthedocs.io/en/latest/Grobid-service/#service-checks

    Parameters
    ----------
    host
        Host of the GROBID server.
    port
        Port of the GROBID server.

    Returns
    -------
    bool
        Whether the GROBID server is alive.
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/isalive")
    except requests.RequestException:
        return False

    if response.content == b"true":
        return True
    else:
        return False


def grobid_pdf_to_tei_xml(pdf_content: bytes, host: str, port: int) -> str:
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
    return asyncio.run(convert_single(pdf_content, host, port))


async def convert_single(pdf_content, host, port):
    async with aiohttp.ClientSession() as session:
        xml_content = await grobid_pdf_to_tei_xml_aio(pdf_content, host, port, session)

    return xml_content


async def convert_and_save(pdf_content, xml_path, host, port, session):
    # Conversion
    logger.info(f"Using GROBID at {host}:{port} to convert PDF")
    xml_content = await grobid_pdf_to_tei_xml_aio(pdf_content, host, port, session)

    # Write XML file
    logger.info(f"Writing XML file to {xml_path.resolve().as_uri()}")
    with xml_path.open("w") as fh:
        fh.write(xml_content)


async def grobid_pdf_to_tei_xml_aio(
    pdf_content: bytes,
    host: str,
    port: int,
    session
) -> str:
    url = f"http://{host}:{port}/api/processFulltextDocument"
    files = {"input": pdf_content}
    headers = {"Accept": "application/xml"}
    async with session.post(url, data=files, headers=headers) as response:
        xml_content = await response.text()

    return xml_content

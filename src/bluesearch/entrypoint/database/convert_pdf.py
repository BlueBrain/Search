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
"""Implementation of the convert-pdf subcommand."""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import textwrap
from typing import Iterable

import aiohttp

from bluesearch.database.pdf import grobid_is_alive, grobid_pdf_to_tei_xml, grobid_pdf_to_tei_xml_aio

logger = logging.getLogger(__name__)


def init_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initialise the argument parser for the convert-pdf subcommand.

    Parameters
    ----------
    parser
        The argument parser to initialise.

    Returns
    -------
    argparse.ArgumentParser
        The initialised argument parser. The same object as the `parser`
        argument.
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    description = """
    Parse a PDF file using the GROBID service and produce a TEI XML
    file. It's assumed that the GROBID service is running under the
    host/port combination provided.

    For more information on how to host such a service refer to the official
    documentation: https://grobid.readthedocs.io/en/latest/Grobid-docker
    """
    parser.description = textwrap.dedent(description)

    parser.add_argument(
        "grobid_host",
        type=str,
        metavar="GROBID-HOST",
        help="The host of the GROBID server.",
    )
    parser.add_argument(
        "grobid_port",
        type=int,
        metavar="GROBID-PORT",
        help="The port of the GROBID server.",
    )
    parser.add_argument(
        "input_path",
        type=pathlib.Path,
        metavar="INPUT-PATH",
        help="The path to a single PDF file or a directory with many PDF files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        metavar="OUTPUT-DIR",
        help="""
        The output directory where the XML file(s) will be saved. If not
        provided the output files will be placed in the same directory as
        the input files.
        """,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="""
        Overwrite the output files if they already exits. Without this flag
        all PDF files for which the output XML file already exists will
        be skipped
        """,
    )

    return parser


def run(
    grobid_host: str,
    grobid_port: int,
    input_path: pathlib.Path,
    output_dir: pathlib.Path | None,
    *,
    force: bool,
) -> int:
    """Run the convert-pdf subcommand.

    Note that the names and types of the parameters should match the parser
    arguments added in ``init_parser``. The purpose of the matching is to be
    able to combine the functions in this way:

    >>> import argparse
    >>> parser = init_parser(argparse.ArgumentParser())
    >>> args = parser.parse_args()
    >>> run(**vars(args))

    This will run the convert-pdf subcommand implemented here as a standalone
    application.

    Parameters
    ----------
    grobid_host
        The host of the GROBID service.
    grobid_port
        The port of the GROBID service.
    input_path
        The path to the input PDF file or a directory with PDF files.
    output_dir
        The output directory for the XML files.
    force
        If true overwrite the output file if it already exists.

    Returns
    -------
    int
        The exit code of the command
    """
    # Check the GROBID server
    if not grobid_is_alive(grobid_host, grobid_port):
        logger.error("The GROBID server is not alive")
        return 1

    # Check if the input file exists
    if not input_path.exists():
        logger.error(f"The input path {str(input_path)!r} does not exist")
        return 1

    # Collect input paths
    input_paths: Iterable[pathlib.Path]
    if input_path.is_file():
        input_paths = [input_path]
    else:
        input_paths = _keep_pdfs_only(input_path.iterdir())

    # Convert
    path_map = prepare_files(input_paths, output_dir, force)
    if len(path_map) > 0:
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Starting asynchronous PDF conversion: {len(path_map)} files")
        asyncio.run(_convert_and_save_all(path_map, grobid_host, grobid_port))
    else:
        logger.info("No files to convert")

    return 0


def _keep_pdfs_only(pdf_paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    """Filter a collection of paths and paths to PDF files only.

    Parameters
    ----------
    pdf_paths
        An iterable of paths.

    Returns
    -------
    A list of paths that are files and have the PDF file extension.
    """
    filtered_paths = []
    for path in pdf_paths:
        if not path.is_file():
            logger.info("Will skip directory %s", path.resolve().as_uri())
            continue
        if path.suffix.lower() != ".pdf":
            logger.info("Will skip non-PDF file %s", path.resolve().as_uri())
            continue
        filtered_paths.append(path)

    return filtered_paths


def prepare_files(input_paths, output_dir, force):
    path_map = {}
    for pdf_path in input_paths:
        output_name = pdf_path.with_suffix(".xml").name
        if output_dir is None:
            output_path = pdf_path.parent / output_name
        else:
            output_path = output_dir / output_name

        if output_path.exists() and not force:
            logger.info(
                "Not overwriting existing file %s, use --force to always overwrite.",
                output_path.resolve().as_uri(),
            )
        else:
            path_map[pdf_path] = output_path

    return path_map


async def _convert_and_save_all(path_map, grobid_host, grobid_port):
    async with aiohttp.ClientSession() as session:
        tasks = [
            convert_and_save(pdf_path, xml_path, grobid_host, grobid_port, session)
            for pdf_path, xml_path in path_map.items()
        ]

        await asyncio.gather(*tasks)


async def convert_and_save(pdf_path, xml_path, grobid_host, grobid_port, session):
    logger.info(f"Reading {pdf_path.resolve().as_uri()}")
    with pdf_path.open("rb") as fh_pdf:
        pdf_content = fh_pdf.read()

    logger.info(f"Converting {pdf_path.resolve().as_uri()} to XML")
    xml_content = await grobid_pdf_to_tei_xml_aio(pdf_content, grobid_host, grobid_port, session)

    with xml_path.open("w") as fh_xml:
        n_bytes = fh_xml.write(xml_content)
    logger.info(f"Wrote {xml_path.resolve().as_uri()} to disk ({n_bytes:,d} bytes)")

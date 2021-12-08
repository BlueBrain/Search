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
import logging
import pathlib
import textwrap
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from bluesearch.database.pdf import grobid_is_alive, grobid_pdf_to_tei_xml

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
        help="""
        The path to a single PDF file or a directory with many PDF files. In
        the latter case all files with the extension ".pdf" will be globbed
        recursively in all subdirectories.
        """,
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
        "-w",
        "--num-workers",
        type=int,
        default=5,
        help="The number of workers",
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
    num_workers,
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
    num_workers
        The number of parallel workers.
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
        input_paths = input_path.rglob("*.pdf")

    failed_paths = []
    path_map = _prepare_output_paths(input_paths, output_dir, force)

    if len(path_map) == 0:
        logger.warning("No files to process, stopping")
        return 0

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    # Convert
    def do_work(pdf_path):
        xml_path = path_map[pdf_path]
        try:
            _convert_pdf_file(grobid_host, grobid_port, pdf_path, xml_path)
        except Exception as exc:
            logger.exception(
                f"An error happened when processing {pdf_path.resolve().as_uri()}: "
                f"{exc}"
            )
            # lists seem to be thread-safe in Python
            failed_paths.append(pdf_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(do_work, path_map, timeout=60)

    for path in failed_paths:
        logger.warning(f"Failed to process {path.resolve().as_uri()}")

    return 0


def _prepare_output_paths(
    input_paths: Iterable[pathlib.Path],
    output_dir: pathlib.Path | None,
    force: bool,
) -> dict[pathlib.Path, pathlib.Path]:
    """Assign output XML paths to all input PDF paths.

    Parameters
    ----------
    input_paths
        A sequence of input PDF paths.
    output_dir
        The output directory. If None then the output files will be placed
        in the same respective directory as the input PDF.
    force
        If False then the PDFs for which the outputs already exist will be
        skipped.

    Returns
    -------
    dict
        A mapping from input paths to output paths.
    """
    path_map = {}

    for input_path in input_paths:
        output_name = input_path.with_suffix(".xml").name
        if output_dir is None:
            output_path = input_path.parent / output_name
        else:
            output_path = output_dir / output_name

        if output_path.exists() and not force:
            logger.warning(
                "Not overwriting existing file %s, use --force to always overwrite.",
                output_path.resolve().as_uri(),
            )
        else:
            path_map[input_path] = output_path

    return path_map


def _convert_pdf_file(
    grobid_host: str,
    grobid_port: int,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Convert a single PDF file to XML and write the XML file to disk.

    Parameters
    ----------
    grobid_host
        The host of the GROBID service.
    grobid_port
        The port of the GROBID service.
    input_path
        The path to the input PDF file.
    output_path
        The output directory for the XML file.
    """
    logger.info(f"Reading {input_path.resolve().as_uri()}")
    with input_path.open("rb") as fh_pdf:
        pdf_content = fh_pdf.read()

    logger.info(f"Converting {input_path.resolve().as_uri()} to XML")
    xml_content = grobid_pdf_to_tei_xml(pdf_content, grobid_host, grobid_port)

    with output_path.open("w") as fh_xml:
        n_bytes = fh_xml.write(xml_content)
    logger.info(f"Wrote {output_path.resolve().as_uri()} ({n_bytes:,d} bytes)")

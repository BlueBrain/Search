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
        "input_pdf_path",
        type=pathlib.Path,
        metavar="INPUT-PDF-PATH",
        help="The path of the input PDF file.",
    )
    parser.add_argument(
        "output_xml_path",
        type=pathlib.Path,
        metavar="OUTPUT-XML-PATH",
        help="The path of the output XML file.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite the output file if it already exits.",
    )

    return parser


def run(
    grobid_host: str,
    grobid_port: int,
    input_pdf_path: pathlib.Path,
    output_xml_path: pathlib.Path,
    *,
    force: bool,
) -> int:
    """Run the convert-pdf subcommand.

    Note that the names and types of the parameters should match the parser
    arguments added in ``init_parser``. The purpose of the matching is to be
    able to combine the functions in this way:

    >>> import argparse
    >>> from bluesearch.entrypoint.database import convert_pdf
    >>> parser = convert_pdf.init_parser(argparse.ArgumentParser())
    >>> # replace with true values and uncomment
    >>> argv = ["host", "port", "pdf_path", "xml_path"]
    >>> # args = parser.parse_args(argv)
    >>> # convert_pdf.run(**vars(args))

    This will run the convert-pdf subcommand implemented here as a standalone
    application.

    Parameters
    ----------
    grobid_host
        The host of the GROBID service.
    grobid_port
        The port of the GROBID service.
    input_pdf_path
        The path to the input PDF file.
    output_xml_path
        The path to the output XML file.
    force
        If true overwrite the output file if it already exists.

    Returns
    -------
    int
        The exit code of the command
    """
    # Check if the input file exists
    if not input_pdf_path.exists():
        logger.error(
            f"The input file {str(input_pdf_path)!r} does not exist.",
        )
        return 1

    # Check if the output file already exists
    if output_xml_path.exists() and not force:
        logger.error(
            f"The output file {str(output_xml_path)!r} already exists. "
            "Either delete it or use the --force option to overwrite it.",
        )
        return 1

    # Read the PDF file
    logger.info("Reading the PDF file")
    with input_pdf_path.open("rb") as fh_pdf:
        pdf_content = fh_pdf.read()

    # Convert the PDF to XML
    logger.info("Converting PDF to XML")
    from bluesearch.database.pdf import grobid_pdf_to_tei_xml

    xml_content = grobid_pdf_to_tei_xml(pdf_content, grobid_host, grobid_port)

    # Write the XML file
    logger.info("Writing the XML file to disk")
    with output_xml_path.open("w") as fh_xml:
        n_bytes = fh_xml.write(xml_content)
    logger.info("Wrote %d bytes to %s", n_bytes, output_xml_path.resolve().as_uri())

    logger.info("PDF conversion done")

    return 0

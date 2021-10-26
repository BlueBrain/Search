"""Implementation of the convert-pdf subcommand."""
from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from typing import Sequence, Text


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
    Parse a PDF file using and the GROBID service and produce a TEI XML
    file. It's assumed that the GROBID service is running under the
    host/port combination provided.

    For more information on how to host such a survice refer to the official
    documentation: https://grobid.readthedocs.io/en/latest/Grobid-docker
    """
    parser.description = textwrap.dedent(description)

    parser.add_argument(
        "grobid_host",
        type=str,
        metavar="GROBID-HOST",
        help="The host of the GROBID server."
    )
    parser.add_argument(
        "grobid_port",
        type=int,
        metavar="GROBID-PORT",
        help="The port of the GROBID server."
    )
    parser.add_argument(
        "input_pdf_path",
        type=pathlib.Path,
        metavar="INPUT-PDF-PATH",
        help="The path of the input PDF file."
    )
    parser.add_argument(
        "output_xml_path",
        type=pathlib.Path,
        metavar="OUTPUT-XML-PATH",
        help="The path of the output XML file."
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite the output file if it already exits."
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
    >>> parser = init_parser(argparse.ArgumentParser())
    >>> args = parser.parse_args()
    >>> run(**vars(args))

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
    print("host:", grobid_host, type(grobid_host))
    print("port:", grobid_port, type(grobid_port))
    print("pdf path:", input_pdf_path, type(input_pdf_path))
    print("output path:", output_xml_path, type(output_xml_path))
    print("force:", force, type(force))

    # Check if the input file exists
    if not input_pdf_path.exists():
        print(
            f"ERROR: The input file {str(input_pdf_path)!r} does not exist.",
            file=sys.stderr
        )
        return 1

    # Check if the output file already exists
    if output_xml_path.exists() and not force:
        print(
            f"ERROR: The output file {str(output_xml_path)!r} already exists. "
            "Either delete it or use the --force option to overwrite it.",
            file=sys.stderr
        )
        return 1

    from bluesearch.database.pdf import grobid_pdf_to_tei_xml

    return 0


def main(argv: Sequence[Text] | None = None) -> int:
    """Run the convert-xml command as a standalone application.

    Parameters
    ----------
    argv
        The argument vector.

    Returns
    -------
    int
        The exit code of the program.
    """
    parser = init_parser(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    return run(**vars(args))


if __name__ == "__main__":
    sys.exit(main())

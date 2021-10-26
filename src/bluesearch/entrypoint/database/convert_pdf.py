"""Implementation of the convert-pdf subcommand."""
from __future__ import annotations

import argparse
import pathlib
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

    parser.add_argument("grobid_host", metavar="GROBID-HOST")
    parser.add_argument("grobid_port", type=int, metavar="GROBID-PORT")
    parser.add_argument("input_pdf_path", metavar="INPUT-PDF-PATH")
    parser.add_argument("output_xml_path", metavar="OUTPUT-XML-PATH")

    return parser


def run(
    grobid_host: str,
    grobid_port: int,
    input_pdf_path: pathlib.Path,
    output_xml_path: pathlib.Path,
) -> None:
    """Run the convert-pdf subcommand.

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
    """
    print("host:", grobid_host)
    print("port:", grobid_port)
    print("pdf path:", input_pdf_path)
    print("output path:", output_xml_path)


def main(argv: Sequence[Text] | None = None) -> None:
    """Run the convert-xml command as a standalone application.

    Parameters
    ----------
    argv
        The argument vector.
    """
    parser = init_parser(argparse.ArgumentParser())
    args = parser.parse_args(argv)
    run(**vars(args))


if __name__ == "__main__":
    main()

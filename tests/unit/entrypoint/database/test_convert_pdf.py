import argparse
import inspect
import pathlib
import unittest.mock

import pytest

from bluesearch.entrypoint.database import convert_pdf

CONVERT_PDF_PARAMS = {
    "grobid_host",
    "grobid_port",
    "input_pdf_path",
    "output_xml_path",
    "force",
}


def test_init_parser():
    parser = convert_pdf.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(
        ["my-host", "1234", "/path/to/file.pdf", "/path/to/file.xml"]
    )
    assert vars(args).keys() == CONVERT_PDF_PARAMS

    # Test the values
    assert args.grobid_host == "my-host"
    assert args.grobid_port == 1234
    assert args.input_pdf_path == pathlib.Path("/path/to/file.pdf")
    assert args.output_xml_path == pathlib.Path("/path/to/file.xml")
    assert args.force is False


def test_run_has_consistent_parameters():
    assert inspect.signature(convert_pdf.run).parameters.keys() == CONVERT_PDF_PARAMS


class TestRun:
    @pytest.mark.parametrize(
        ("is_alive", "expected_exit_code"),
        (
            (True, 0),
            (False, 1),
        ),
    )
    @unittest.mock.patch("bluesearch.database.pdf.grobid_pdf_to_tei_xml")
    @unittest.mock.patch("bluesearch.database.pdf.grobid_is_alive")
    def test_is_alive_check_works(
        self,
        grobid_is_alive,
        grobid_pdf_to_tei_xml,
        is_alive,
        expected_exit_code,
        tmp_path,
    ):
        # Configure mocks
        grobid_pdf_to_tei_xml.return_value = "<xml>parsed</xml>"
        grobid_is_alive.return_value = is_alive

        # Configure file paths
        input_pdf_file = tmp_path / "my-file.pdf"
        input_pdf_file.touch()
        output_xml_file = tmp_path / "my-file.xml"

        # Run the test
        exit_code = convert_pdf.run(
            "host",
            1234,
            input_pdf_file,
            output_xml_file,
            force=False,
        )
        assert exit_code == expected_exit_code

    @unittest.mock.patch("bluesearch.database.pdf.grobid_is_alive")
    def test_nonexistent_pdf_errors(self, grobid_is_alive):
        grobid_is_alive.return_value = True

        input_pdf_file = pathlib.Path("/a/nonexistent/file.pdf")
        assert not input_pdf_file.exists()
        output_xml_file = pathlib.Path("/does/not/matter.xml")

        exit_code = convert_pdf.run(
            "host", 1234, input_pdf_file, output_xml_file, force=False
        )
        assert exit_code == 1

    @unittest.mock.patch("bluesearch.database.pdf.grobid_is_alive")
    def test_output_file_exists_errors(self, grobid_is_alive, tmp_path):
        grobid_is_alive.return_value = True

        input_pdf_file = tmp_path / "my-file.pdf"
        input_pdf_file.touch()
        output_xml_file = tmp_path / "my-file.xml"
        output_xml_file.touch()

        exit_code = convert_pdf.run(
            "host", 1234, input_pdf_file, output_xml_file, force=False
        )
        assert exit_code == 1

    @unittest.mock.patch("bluesearch.database.pdf.grobid_pdf_to_tei_xml")
    @unittest.mock.patch("bluesearch.database.pdf.grobid_is_alive")
    def test_pdf_conversion_works(
        self, grobid_is_alive, grobid_pdf_to_tei_xml, tmp_path
    ):
        grobid_is_alive.return_value = True

        # Prepare the input PDF file
        input_pdf_file = tmp_path / "my-file.pdf"
        with input_pdf_file.open("wb") as fh:
            fh.write(b"PDF file content")

        # Prepare the output XML file path
        output_xml_file = tmp_path / "my-file.xml"

        # Set up the mock
        grobid_pdf_to_tei_xml.return_value = "<xml>parsed</xml>"

        # Call the entry point
        exit_code = convert_pdf.run(
            "host", 1234, input_pdf_file, output_xml_file, force=False
        )

        # Checks
        assert exit_code == 0
        grobid_pdf_to_tei_xml.assert_called_once()
        grobid_pdf_to_tei_xml.assert_called_with(b"PDF file content", "host", 1234)
        with output_xml_file.open() as fh:
            assert fh.read() == "<xml>parsed</xml>"

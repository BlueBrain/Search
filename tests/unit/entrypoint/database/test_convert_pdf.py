import argparse
import inspect
import logging
import pathlib
import unittest.mock

import pytest

from bluesearch.entrypoint.database import convert_pdf

CONVERT_PDF_PARAMS = {
    "grobid_host",
    "grobid_port",
    "input_path",
    "output_dir",
    "num_workers",
    "force",
}


def test_init_parser():
    parser = convert_pdf.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["my-host", "1234", "/input/path"])
    assert vars(args).keys() == CONVERT_PDF_PARAMS

    # Test the values
    assert args.grobid_host == "my-host"
    assert args.grobid_port == 1234
    assert args.input_path == pathlib.Path("/input/path")
    assert args.output_dir is None
    assert args.num_workers > 0
    assert args.force is False


def test_run_has_consistent_parameters():
    assert inspect.signature(convert_pdf.run).parameters.keys() == CONVERT_PDF_PARAMS


class TestPrepareOutputPaths:
    def test_output_same_folder(self):
        input_paths = [
            pathlib.Path("/a/b/x.pdf"),
            pathlib.Path("/c/d/y.pdf"),
        ]
        path_map = convert_pdf._prepare_output_paths(input_paths, None, force=False)
        assert set(path_map) == set(input_paths)
        for input_path, output_path in path_map.items():
            assert output_path == input_path.with_suffix(".xml")

    def test_output_fixed_folder(self):
        output_folder = pathlib.Path("/output")
        input_paths = [
            pathlib.Path("/a/b/x.pdf"),
            pathlib.Path("/c/d/y.pdf"),
        ]
        expected_output_paths = [
            output_folder / "x.xml",
            output_folder / "y.xml",
        ]
        path_map = convert_pdf._prepare_output_paths(
            input_paths, output_folder, force=False
        )
        assert set(path_map) == set(input_paths)
        assert list(path_map.values()) == expected_output_paths

    def test_existing_files_are_skipped(self, tmp_path, caplog):
        input_path = tmp_path / "x.pdf"
        output_xml = tmp_path / "x.xml"
        output_xml.touch()
        with caplog.at_level(logging.WARNING):
            path_map = convert_pdf._prepare_output_paths(
                [input_path], None, force=False
            )
        assert len(path_map) == 0
        assert "Not overwriting" in caplog.text

    def test_force_option_works(self, tmp_path):
        input_path = tmp_path / "x.pdf"
        output_xml = tmp_path / "x.xml"

        # Output file exists, but should be overwritten because of force=True
        output_xml.touch()
        path_map = convert_pdf._prepare_output_paths([input_path], None, force=True)
        assert len(path_map) == 1


class TestRun:
    @pytest.mark.parametrize(
        ("is_alive", "expected_exit_code"),
        (
            (True, 0),
            (False, 1),
        ),
    )
    @unittest.mock.patch(
        "bluesearch.entrypoint.database.convert_pdf.grobid_pdf_to_tei_xml"
    )
    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
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

        # Run the test
        exit_code = convert_pdf.run(
            "host",
            1234,
            input_pdf_file,
            None,
            num_workers=1,
            force=False,
        )
        assert exit_code == expected_exit_code

    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
    def test_nonexistent_pdf_errors(self, grobid_is_alive):
        grobid_is_alive.return_value = True

        input_pdf_file = pathlib.Path("/a/nonexistent/file.pdf")
        assert not input_pdf_file.exists()
        output_xml_file = pathlib.Path("/does/not/matter.xml")

        exit_code = convert_pdf.run(
            "host", 1234, input_pdf_file, output_xml_file, num_workers=1, force=False
        )
        assert exit_code == 1

    @unittest.mock.patch(
        "bluesearch.entrypoint.database.convert_pdf.grobid_pdf_to_tei_xml"
    )
    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
    def test_pdf_conversion_works_input_file(
        self, grobid_is_alive, grobid_pdf_to_tei_xml, tmp_path
    ):
        grobid_is_alive.return_value = True
        output_dir = tmp_path / "output"

        # Prepare the input PDF file
        input_pdf_file = tmp_path / "my-file.pdf"
        with input_pdf_file.open("wb") as fh:
            fh.write(b"PDF file content")

        # Prepare the output XML file path
        output_xml_file = output_dir / "my-file.xml"

        # Set up the mock
        grobid_pdf_to_tei_xml.return_value = "<xml>parsed</xml>"

        # Call the entry point
        exit_code = convert_pdf.run(
            "host", 1234, input_pdf_file, output_dir, num_workers=1, force=False
        )

        # Checks
        assert exit_code == 0
        grobid_pdf_to_tei_xml.assert_called_once()
        grobid_pdf_to_tei_xml.assert_called_with(b"PDF file content", "host", 1234)
        with output_xml_file.open() as fh:
            assert fh.read() == "<xml>parsed</xml>"

    @unittest.mock.patch(
        "bluesearch.entrypoint.database.convert_pdf.grobid_pdf_to_tei_xml"
    )
    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
    def test_pdf_conversion_exception(
        self,
        grobid_is_alive,
        grobid_pdf_to_tei_xml,
        tmp_path,
        caplog,
    ):
        grobid_is_alive.return_value = True

        # Prepare the input PDF file
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        # Good content PDF
        input_pdf_file_good = input_dir / "good.pdf"
        with input_pdf_file_good.open("wb") as fh:
            fh.write(b"PDF file content good")
        output_xml_file_good = input_dir / "good.xml"  # will exist

        # Bad content PDF
        input_pdf_file_bad = input_dir / "bad.pdf"
        with input_pdf_file_bad.open("wb") as fh:
            fh.write(b"PDF file content bad")
        output_xml_file_bad = input_dir / "bad.xml"  # will not exist

        # Set up the mock
        def fake_grobid_pdf_to_tei_xml(pdf_content, grobid_host, grobid_port):
            pdf_content_str = str(pdf_content)

            if "bad" in pdf_content_str:
                raise ValueError

            return "<xml>parsed</xml>"

        grobid_pdf_to_tei_xml.side_effect = fake_grobid_pdf_to_tei_xml

        # Call the entry point
        exit_code = convert_pdf.run(
            "host", 1234, input_dir, None, num_workers=1, force=False
        )

        # Checks
        assert exit_code == 0
        assert grobid_pdf_to_tei_xml.call_count == 2

        # Check the converted XMLs
        assert output_xml_file_good.exists()
        assert not output_xml_file_bad.exists()
        with output_xml_file_good.open() as fh:
            assert fh.read() == "<xml>parsed</xml>"

        # Check failed log
        assert str(input_pdf_file_bad) in caplog.text
        assert str(input_pdf_file_good) not in caplog.text

    @unittest.mock.patch(
        "bluesearch.entrypoint.database.convert_pdf.grobid_pdf_to_tei_xml"
    )
    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
    def test_pdf_conversion_works_input_dir(
        self, grobid_is_alive, grobid_pdf_to_tei_xml, tmp_path
    ):
        grobid_is_alive.return_value = True

        # Prepare the input PDF file
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        output_xml_files = []
        for i in range(3):
            input_pdf_file = input_dir / f"my-file-{i}.pdf"
            with input_pdf_file.open("wb") as fh:
                fh.write(b"PDF file content")
            output_xml_files.append(input_dir / f"my-file-{i}.xml")

        # Set up the mock
        grobid_pdf_to_tei_xml.return_value = "<xml>parsed</xml>"

        # Call the entry point
        exit_code = convert_pdf.run(
            "host", 1234, input_dir, None, num_workers=1, force=False
        )

        # Checks
        assert exit_code == 0
        assert grobid_pdf_to_tei_xml.call_count == 3
        grobid_pdf_to_tei_xml.assert_called_with(b"PDF file content", "host", 1234)

        for output_xml_file in output_xml_files:
            with output_xml_file.open() as fh:
                assert fh.read() == "<xml>parsed</xml>"

    @unittest.mock.patch("bluesearch.entrypoint.database.convert_pdf.grobid_is_alive")
    def test_pdf_conversion_empty_dir(self, grobid_is_alive, tmp_path, caplog):
        grobid_is_alive.return_value = True
        exit_code = convert_pdf.run(
            "host", 1234, tmp_path, None, num_workers=1, force=False
        )

        assert exit_code == 0
        assert "No files to process, stopping" in caplog.text

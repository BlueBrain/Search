#  Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
#  Copyright (C) 2022 Blue Brain Project, EPFL.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
import argparse
import gzip
import inspect
import json
import pathlib
import textwrap

from bluesearch.entrypoint.database import parse_mesh_rdf

PARSE_MESH_RDF_PARAMS = {
    "mesh_nt_gz_file",
    "output_json_file",
}


def test_init_parser():
    parser = parse_mesh_rdf.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["mesh.nt.gz", "mesh_tree.json"])
    assert vars(args).keys() == PARSE_MESH_RDF_PARAMS

    # Test the values
    assert args.mesh_nt_gz_file == pathlib.Path("mesh.nt.gz")
    assert args.output_json_file == pathlib.Path("mesh_tree.json")


def test_run_has_consistent_parameters():
    parameters = inspect.signature(parse_mesh_rdf.run).parameters.keys()
    assert parameters == PARSE_MESH_RDF_PARAMS


class TestRun:
    def test_invalid_mesh_file(self, caplog, tmp_path):
        mesh_file = tmp_path / "mesh.nt.gz"
        output_file = tmp_path / "mesh_tree.json"

        # File doesn't exist
        exit_code = parse_mesh_rdf.run(
            mesh_nt_gz_file=mesh_file,
            output_json_file=output_file,
        )
        assert exit_code != 0
        assert "does not exist" in caplog.text
        caplog.clear()

        # Path points to a directory
        mesh_file.mkdir()
        exit_code = parse_mesh_rdf.run(
            mesh_nt_gz_file=mesh_file,
            output_json_file=output_file,
        )
        assert exit_code != 0
        assert "must be a file" in caplog.text

    def test_parsing(self, tmp_path):
        mesh_file = tmp_path / "mesh.nt.gz"
        output_file = tmp_path / "mesh_tree.json"

        # Prepare the input MeSH RDF file
        nlm = "http://id.nlm.nih.gov/mesh"
        rdf = "http://www.w3.org/2000/01/rdf-schema"
        data = f"""\
        <{nlm}/2022/D123> <{rdf}#label> "Topic"@en .
        <{nlm}/2022/D123> <{rdf}#label> "Sujet"@fr .
        <{nlm}/2022/D123> <{rdf}#some-tag> some-value .
        <{nlm}/2022/D123> <{nlm}/vocab#treeNumber> <{nlm}/2022/A00.123.456> .
        <{nlm}/2022/A00.123.456> <{nlm}/vocab#parent> <{nlm}/2022/A00.123> .
        """
        with gzip.open(mesh_file, "wt") as fh:
            fh.write(textwrap.dedent(data))

        # Test
        exit_code = parse_mesh_rdf.run(
            mesh_nt_gz_file=mesh_file,
            output_json_file=output_file,
        )
        assert exit_code == 0
        with output_file.open() as fh:
            mesh_tree = json.load(fh)
        assert mesh_tree == {"A00.123.456": "Topic"}

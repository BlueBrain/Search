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
import io
import json
import textwrap

import pytest

from bluesearch.database import mesh


class TestMeSHTree:
    def test_initialisation(self):
        mesh_tree_data = {
            "A0": "root",
            "A0.1": "topic",
            "A0.1.1": "subtopic",
            "A0.1.2": "different subtopic",
            "A0.2": "alternative topic",
            "A0.2.1": "subtopic",
        }
        mesh_tree = mesh.MeSHTree(mesh_tree_data)
        assert mesh_tree.tree_number_to_label == mesh_tree_data
        assert mesh_tree.label_to_tree_numbers == {
            "root": ["A0"],
            "topic": ["A0.1"],
            "alternative topic": ["A0.2"],
            "subtopic": ["A0.1.1", "A0.2.1"],
            "different subtopic": ["A0.1.2"],
        }

    def test_loading_from_disk(self, tmp_path):
        data_path = tmp_path / "mesh_tree.json"
        data_path.write_text(json.dumps({"A0": "topic"}))
        mesh_tree = mesh.MeSHTree.load(data_path)
        assert mesh_tree.tree_number_to_label == {"A0": "topic"}
        assert mesh_tree.label_to_tree_numbers == {"topic": ["A0"]}

    def test_parents(self):
        assert list(mesh.MeSHTree.parents("A0.123.456")) == ["A0.123", "A0"]

    def test_parent_topics(self):
        mesh_tree_data = {
            "A0": "root",
            "A0.1": "topic",
            "A0.1.1": "subtopic",
            "A0.1.2": "different subtopic",
            "A0.2": "alternative topic",
            "A0.2.1": "subtopic",
        }
        mesh_tree = mesh.MeSHTree(mesh_tree_data)
        assert mesh_tree.parent_topics("root") == set()
        assert mesh_tree.parent_topics("different subtopic") == {"topic", "root"}
        assert mesh_tree.parent_topics("subtopic") == {
            "alternative topic",
            "topic",
            "root",
        }


def test_resolve_parents():
    mesh_tree_data = {
        "A0": "root",
        "A0.1": "topic",
        "A0.1.1": "subtopic",
        "A0.1.2": "different subtopic",
        "A0.2": "alternative topic",
        "A0.2.1": "subtopic",
    }
    mesh_tree = mesh.MeSHTree(mesh_tree_data)
    assert mesh.resolve_parents(["root"], mesh_tree) == {"root"}
    assert mesh.resolve_parents(["topic"], mesh_tree) == {"topic", "root"}
    assert mesh.resolve_parents(["topic", "alternative topic"], mesh_tree) == {
        "topic",
        "alternative topic",
        "root",
    }
    assert mesh.resolve_parents(["subtopic", "different subtopic"], mesh_tree) == {
        "subtopic",
        "different subtopic",
        "topic",
        "root",
        "alternative topic",
    }


class TestParseTreeNumbers:
    def test_empty_input(self):
        with io.StringIO("") as stream:
            mesh_tree = mesh.parse_tree_numbers(stream)
        assert mesh_tree == {}

    def test_correct_parsing(self):
        nlm = "http://id.nlm.nih.gov/mesh"
        rdf = "http://www.w3.org/2000/01/rdf-schema"
        data = f"""\
        <{nlm}/2022/D123> <{rdf}#label> "Topic"@en .
        <{nlm}/2022/D123> <{rdf}#label> "Sujet"@fr .
        <{nlm}/2022/D123> <{rdf}#some-tag> some-value .
        <{nlm}/2022/D123> <{nlm}/vocab#treeNumber> <{nlm}/2022/A00.123.456> .
        <{nlm}/2022/A00.123.456> <{nlm}/vocab#parent> <{nlm}/2022/A00.123> .
        """
        with io.StringIO(textwrap.dedent(data)) as stream:
            mesh_tree = mesh.parse_tree_numbers(stream)
        assert mesh_tree == {"A00.123.456": "Topic"}

    def test_invalid_triple(self):
        with io.StringIO("not-a-triple") as stream:
            with pytest.raises(RuntimeError, match="not a valid triple"):
                mesh.parse_tree_numbers(stream)

    def test_label_clash(self):
        nlm = "http://id.nlm.nih.gov/mesh"
        rdf = "http://www.w3.org/2000/01/rdf-schema"
        data = f"""\
        <{nlm}/2022/D123> <{rdf}#label> "Topic"@en .
        <{nlm}/2022/D123> <{rdf}#label> "Other topic"@en .
        <{nlm}/2022/D123> <{nlm}/vocab#treeNumber> <{nlm}/2022/A00.123.456> .
        """
        with io.StringIO(textwrap.dedent(data)) as stream:
            with pytest.raises(RuntimeError, match=r"(?i)multiple labels"):
                mesh.parse_tree_numbers(stream)

    def test_invalid_tree_number(self):
        nlm = "http://id.nlm.nih.gov/mesh"
        rdf = "http://www.w3.org/2000/01/rdf-schema"
        data = f"""\
        <{nlm}/2022/D123> <{rdf}#label> "Topic"@en .
        <{nlm}/2022/D123> <{nlm}/vocab#treeNumber> <invalid-format> .
        """
        with io.StringIO(textwrap.dedent(data)) as stream:
            with pytest.raises(RuntimeError, match=r"(?i)cannot parse tree number"):
                mesh.parse_tree_numbers(stream)

    def test_duplicate_tree_number(self):
        nlm = "http://id.nlm.nih.gov/mesh"
        rdf = "http://www.w3.org/2000/01/rdf-schema"
        data = f"""\
        <{nlm}/2022/D123> <{rdf}#label> "Topic 1"@en .
        <{nlm}/2022/D123> <{nlm}/vocab#treeNumber> <{nlm}/2022/A00.123.456> .
        <{nlm}/2022/D456> <{rdf}#label> "Topic 2"@en .
        <{nlm}/2022/D456> <{nlm}/vocab#treeNumber> <{nlm}/2022/A00.123.456> .
        """
        with io.StringIO(textwrap.dedent(data)) as stream:
            with pytest.raises(RuntimeError, match=r"(?i)duplicate tree number"):
                mesh.parse_tree_numbers(stream)

import json
from argparse import ArgumentError
from pathlib import Path

import pytest

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main
from bluesearch.entrypoint.database.parse import iter_parsers


@pytest.mark.parametrize(
    "input_type, path, article_uids",
    [
        pytest.param(
            "cord19-json",
            "cord19_v35/document_parses/pmc_json/PMC7186928.xml.json",
            ["990d33f52ed226346404cff23111989d"],
            id="cord19-json",
        ),
        pytest.param(
            "jats-xml",
            "jats_article.xml",
            ["34eaed1a1a05166c0b8610336aee638d"],
            id="jats-xml",
        ),
        pytest.param(
            "pubmed-xml",
            "pubmed_article.xml",
            ["0e8400416a385b9a62d8178539b76daf"],
            id="pubmed-xml",
        ),
        pytest.param(
            "pubmed-xml-set",
            "pubmed_articles.xml",
            ["e9bb8ba085982a7cbb7d9ac2dbbafc7f", "49442b9ec575ae01f4934dfd79d03631"],
            id="pubmed-xml-set",
        ),
        pytest.param(
            "tei-xml",
            "1411.7903v4.xml",
            ["73604b8751f2f4ecf63a5cefd042f6a3"],
            id="tei-xml",
        ),
        pytest.param(
            "tei-xml-arxiv",
            "1411.7903v4.xml",
            ["26f61b81976907d1fa5b779511fb1360"],
            id="tei-xml-arxiv",
        )
    ],
)
def test_iter_parsers(input_type, path, article_uids):
    input_path = Path("tests/data/") / path
    parsers = iter_parsers(input_type, input_path)
    for parser, uid in zip(parsers, article_uids):
        assert parser.uid == uid


def test_unknown_input_type():
    wrong_type = "wrong-type"

    with pytest.raises(SystemExit) as exc_info:
        main(["parse", wrong_type, "path_to_input", "path_to_output"])

    # argparse exists with error 2, so we need to "unpack" the exception
    exc = exc_info.value
    context = exc.__context__
    assert exc.code == 2
    assert isinstance(context, ArgumentError)
    assert f"invalid choice: '{wrong_type}'" in str(context)


def test_cord19_json(jsons_path, tmp_path):
    path_to_json = jsons_path / "document_parses" / "pmc_json"
    json_files = sorted(path_to_json.glob("*.json"))
    assert len(json_files) > 0

    # Test parsing single file
    for i, inp_file in enumerate(json_files):
        out_dir = tmp_path / str(i)
        args_and_opts = [
            "parse",
            "cord19-json",
            str(inp_file),
            str(out_dir),
        ]
        main(args_and_opts)
        out_files = list(out_dir.glob("*"))

        assert len(out_files) == 1

        with out_files[0].open() as f:
            data = json.load(f)
            uid = data["uid"]
            assert out_files[0].name == f"{uid}.json"

        serialized = out_files[0].read_text("utf-8")
        loaded_article = Article.from_json(serialized)
        assert isinstance(loaded_article, Article)

    # Test parsing multiple files
    out_dir = tmp_path / "all"
    args_and_opts = [
        "parse",
        "cord19-json",
        str(path_to_json),
        str(out_dir),
    ]
    main(args_and_opts)
    out_files = sorted(out_dir.glob("*"))

    assert len(out_files) == len(json_files)

    for out_file in out_files:
        with out_file.open() as f:
            data = json.load(f)
            uid = data["uid"]
            assert out_file.name == f"{uid}.json"

        serialized = out_file.read_text("utf-8")
        loaded_article = Article.from_json(serialized)
        assert isinstance(loaded_article, Article)

    # Test parsing something that doesn't exist
    with pytest.raises(ValueError):
        args_and_opts = [
            "parse",
            "cord19-json",
            str(path_to_json / "dir_that_does_not_exists"),
            str(out_dir),
        ]
        main(args_and_opts)


def test_pubmed_xml_set(tmp_path):
    input_path = "tests/data/pubmed_articles.xml"
    main(["parse", "pubmed-xml-set", input_path, str(tmp_path)])
    files = sorted(tmp_path.iterdir())
    assert len(files) == 2

    uids = ["49442b9ec575ae01f4934dfd79d03631", "e9bb8ba085982a7cbb7d9ac2dbbafc7f"]
    for file, uid in zip(files, uids):
        assert file.name == f"{uid}.json"
        with file.open() as f:
            data = json.load(f)
            loaded_uid = data["uid"]
            assert loaded_uid == uid


def test_dry_run(capsys):
    input_path = "tests/data/cord19_v35/"
    main(["parse", "cord19-json", input_path, "parsed/", "--dry-run"])
    captured = capsys.readouterr()
    assert captured.out == "tests/data/cord19_v35/metadata.csv\n"


def test_recursive(tmp_path):
    input_path = "tests/data/cord19_v35/document_parses/pdf_json/"
    main(["parse", "cord19-json", input_path, str(tmp_path), "--recursive"])
    filenames = sorted(x.name for x in tmp_path.iterdir())
    expected = [
        "61ba28becef4945b919562ac76349af7.json",
        "84ee8e7458ede952bbb567b06c34fdb2.json",
    ]
    assert filenames == expected


def test_filtering(tmp_path):
    input_path = "tests/data/cord19_v35/"
    options = ["--recursive", "--match-filename", "[a-z0-9]+\\.json"]
    main(["parse", "cord19-json", input_path, str(tmp_path), *options])
    filenames = sorted(x.name for x in tmp_path.iterdir())
    expected = [
        "61ba28becef4945b919562ac76349af7.json",
        "84ee8e7458ede952bbb567b06c34fdb2.json",
    ]
    assert filenames == expected


def test_filtering_empty(tmp_path):
    message = "Value for argument 'match-filename' should not be empty!"
    input_path = "tests/data/cord19_v35/"
    options = ["--recursive", "--match-filename", ""]
    with pytest.raises(ValueError, match=message):
        main(["parse", "cord19-json", input_path, str(tmp_path), *options])

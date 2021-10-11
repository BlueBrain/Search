from argparse import ArgumentError

import pytest

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main


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
        assert out_files[0].name == inp_file.stem + ".json"

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

    for inp_file, out_file in zip(json_files, out_files):
        assert out_file.name == inp_file.stem + ".json"

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

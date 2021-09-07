import pathlib
import pickle
from argparse import ArgumentError

import pytest

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main


def test_unknown_article_type():
    wrong_type = "wrong-type"

    with pytest.raises(SystemExit) as exc_info:
        main(["parse", wrong_type, "path_to_input", "path_to_output"])

    # argparse exists with error 2, so we need to "unpack" the exception
    exc = exc_info.value
    context = exc.__context__
    assert exc.code == 2
    assert isinstance(context, ArgumentError)
    assert f"invalid choice: '{wrong_type}'" in context.args[1]


def test_cord19(jsons_path, tmpdir):
    # Create a dummy database
    all_input_paths = sorted(jsons_path.rglob("*.json"))
    output_folder = pathlib.Path(str(tmpdir))

    n_articles = len(all_input_paths)

    for input_path in all_input_paths:
        args_and_opts = [
            "parse",
            "cord19-json",
            str(input_path),
            str(output_folder / input_path.name),
        ]

        main(args_and_opts)

    # Check
    output_paths = list(output_folder.iterdir())
    n_files = len(output_paths)

    assert n_files == n_articles > 0

    for output_path in output_paths:
        with open(output_path, "rb") as f:
            loaded_article = pickle.load(f)
            assert isinstance(loaded_article, Article)

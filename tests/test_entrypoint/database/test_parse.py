import pathlib
import pickle

import pytest
import sqlalchemy

from bluesearch.database.article import Article
from bluesearch.entrypoint.database.parent import main


def test_unknown_parser():
    with pytest.raises(ValueError, match="Unsupported parser"):
        main(["parse", "WrongParser", "path_to_input", "path_to_output"])


def test_cord19(jsons_path, tmpdir):
    # Create a dummy database
    path_jsons = pathlib.Path(__file__).parent.parent.parent / "data" / "cord19_v35"
    all_input_paths = sorted(path_jsons.rglob("*.json"))
    output_folder = pathlib.Path(str(tmpdir))

    n_articles = len(all_input_paths)

    for input_path in all_input_paths:
        args_and_opts = [
            "parse",
            "CORD19ArticleParser",
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

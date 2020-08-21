import pytest

from bbsearch.entrypoints import run_create_mining_cache


def test_help(capsys):
    with pytest.raises(SystemExit) as error:
        run_create_mining_cache(["--help"])
    stdout, stderr = capsys.readouterr()

    assert error.value.code == 0
    assert stdout.startswith("usage:")
    assert stderr == ""

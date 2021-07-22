import subprocess

import pytest

from bluesearch.entrypoint.database.parent import main


@pytest.mark.parametrize("command", ["add", "init"])
def test_commands_work(command):
    subprocess.check_call(["bbs_database", command, "--help"])

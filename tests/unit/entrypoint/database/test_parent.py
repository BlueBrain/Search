import subprocess

import pytest


@pytest.mark.parametrize("command", ["add", "convert-pdf", "init", "parse"])
def test_commands_work(command):
    subprocess.check_call(["bbs_database", command, "--help"])

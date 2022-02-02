import logging
import subprocess

import pytest

from bluesearch.entrypoint.database.parent import _setup_logging


@pytest.mark.parametrize("command", ["add", "convert-pdf", "init", "parse"])
def test_commands_work(command):
    subprocess.run(["bbs_database", command, "--help"], check=True)


def test_setup_logging():
    """This test modifies logging levels globally.

    However, it should not be a big deal since each CLI entrypoint ends up
    calling `_setup_logging`.
    """
    all_loggers = logging.root.manager.loggerDict
    bluesearch_loggers = {
        k: v
        for k, v in all_loggers.items()
        if k.startswith("bluesearch") and isinstance(v, logging.Logger)
    }
    external_loggers = {
        k: v
        for k, v in all_loggers.items()
        if not k.startswith("bluesearch") and isinstance(v, logging.Logger)
    }

    bluesearch_levels_before = {
        name: logger.getEffectiveLevel() for name, logger in bluesearch_loggers.items()
    }
    external_levels_before = {
        name: logger.getEffectiveLevel() for name, logger in external_loggers.items()
    }

    _setup_logging(logging.DEBUG)

    bluesearch_levels_after = {
        name: logger.getEffectiveLevel() for name, logger in bluesearch_loggers.items()
    }
    external_levels_after = {
        name: logger.getEffectiveLevel() for name, logger in external_loggers.items()
    }

    assert bluesearch_levels_before != bluesearch_levels_after
    assert set(bluesearch_levels_after.values()) == {logging.DEBUG}
    assert external_levels_before == external_levels_after

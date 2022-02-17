from __future__ import annotations

import logging
import subprocess

import pytest

from bluesearch.entrypoint.database.parent import _setup_logging


@pytest.mark.parametrize("command", ["add", "convert-pdf", "init", "parse"])
def test_commands_work(command):
    subprocess.run(["bbs_database", command, "--help"], check=True)


def test_setup_logging(caplog):
    def get_levels(loggers: dict[str, logging.Logger]) -> dict[str, int]:
        """Get logging level for each logger."""
        return {name: logger.getEffectiveLevel() for name, logger in loggers.items()}

    caplog.set_level(logging.WARNING, logger="bluesearch")

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

    bluesearch_levels_before = get_levels(bluesearch_loggers)
    external_levels_before = get_levels(external_loggers)

    _setup_logging(logging.DEBUG)

    bluesearch_levels_after = get_levels(bluesearch_loggers)
    external_levels_after = get_levels(external_loggers)

    assert set(bluesearch_levels_before.values()) == {logging.WARNING}
    assert set(bluesearch_levels_after.values()) == {logging.DEBUG}

    assert external_levels_before == external_levels_after

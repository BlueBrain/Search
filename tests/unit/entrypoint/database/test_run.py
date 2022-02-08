# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import datetime
import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest

from bluesearch.entrypoint.database import run
from bluesearch.utils import JSONL

RUN_PARAMS = {
    "source",
    "from_month",
    "filter_config",
}

def test_init_parser():
    parser = run.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["arxiv", "2021-12", "/path/to/config.jsonl"])
    assert vars(args).keys() == RUN_PARAMS

    # Test the values
    assert args.source == "arxiv"
    assert args.from_month == datetime.datetime(2021, 12, 1)
    assert args.filter_config == pathlib.Path("/path/to/config.jsonl")


def test_run_arguments():
    assert inspect.signature(run.run).parameters.keys() == RUN_PARAMS

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
import inspect
import pathlib

from bluesearch.entrypoint.database import topic_filter

TOPIC_FILTER_PARAMS = {
    "extracted_topics",
    "filter_config",
    "output_file",
}


def test_init_parser():
    parser = topic_filter.init_parser(argparse.ArgumentParser())

    args = parser.parse_args(["/path/to/topics", "/path/to/config", "/path/to/output"])
    assert vars(args).keys() == TOPIC_FILTER_PARAMS

    # Test the values
    assert args.extracted_topics == pathlib.Path("/path/to/topics")
    assert args.filter_config == pathlib.Path("/path/to/config")
    assert args.output_file == pathlib.Path("/path/to/output")


def test_run_arguments():
    assert (
        inspect.signature(topic_filter.run).parameters.keys() == TOPIC_FILTER_PARAMS
    )

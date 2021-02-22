"""Benchmark INSERT operations through Pandas with and without transactions."""

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

import numpy as np
import pandas as pd
import pytest as pt
import sqlalchemy

PORT = 9731


@pt.fixture
def data():
    rng = np.random.default_rng(1739)
    numbers = rng.integers(10, size=100000)
    return pd.DataFrame({"column": numbers})


@pt.fixture
def engine():
    return sqlalchemy.create_engine(
        f"mysql+pymysql://root:root@localhost:{PORT}/benchmarks"
    )


def insert_without_transactions(data, engine):
    data.to_sql("without", engine, if_exists="append", index=False)


def insert_with_transactions(data, engine):
    with engine.begin() as con:
        data.to_sql("with", con, if_exists="append", index=False)


def test_insert_without_transactions(benchmark, data, engine):
    benchmark(insert_without_transactions, data, engine)


def test_insert_with_transactions(benchmark, data, engine):
    benchmark(insert_with_transactions, data, engine)

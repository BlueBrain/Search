"""Benchmark INSERT operations through Pandas with and without transactions."""

import numpy as np
import pandas as pd
import pytest as pt
import sqlalchemy

PORT = 9731


@pt.fixture
def data():
    rng = np.random.default_rng(1739)
    numbers = rng.integers(10, size=100000)
    return pd.DataFrame({'column': numbers})


@pt.fixture
def engine():
    return sqlalchemy.create_engine(f'mysql+pymysql://root:root@localhost:{PORT}/benchmarks')


def insert_without_transactions(data, engine):
    data.to_sql('without', engine, if_exists="append", index=False)


def insert_with_transactions(data, engine):
    with engine.begin() as con:
        data.to_sql('with', con, if_exists="append", index=False)


def test_insert_without_transactions(benchmark, data, engine):
    benchmark(insert_without_transactions, data, engine)


def test_insert_with_transactions(benchmark, data, engine):
    benchmark(insert_with_transactions, data, engine)

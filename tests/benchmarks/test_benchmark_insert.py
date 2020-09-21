"""Benchmark INSERT operations through Pandas with and without transactions."""

import pandas as pd
import pytest as pt
import sqlalchemy

PORT = 9731


@pt.fixture
def data():
    return pd.DataFrame({'column': [1, 2, 3]})


@pt.fixture
def engine():
    return sqlalchemy.create_engine(f'mysql+pymysql://root:root@localhost:{PORT}/benchmarks')


def insert_without_transactions(data, engine):
    data.to_sql('without', engine, if_exists="append", index=False)


def insert_with_transactions(data, engine):
    with engine.begin() as con:
        data.to_sql('without', con, if_exists="append", index=False)


def test_insert_without_transactions(benchmark, data, engine):
    result = benchmark(insert_without_transactions, data, engine)
    print(type(result))


def test_insert_with_transactions(benchmark, data, engine):
    result = benchmark(insert_with_transactions, data, engine)
    print(type(result))

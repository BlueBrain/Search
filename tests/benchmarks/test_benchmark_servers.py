import pytest
import requests
import sqlalchemy

MYSQL_USER = 'guest'
MYSQL_PWD = 'guest'
DATABASE_NAME = 'cord19_v35'

DRIVERS = ['mysql+pymysql', 'mysql+mysqldb', 'mysql']
QUERIES = {'date_range': "SELECT article_id FROM articles WHERE "
                         "publish_time BETWEEN '1999-01-01' AND '2020-12-31'"}


@pytest.mark.parametrize('server_name', ['embedding', 'mining', 'search'])
def test_flask_help(benchmark, benchmark_parameters, server_name):
    server = benchmark_parameters[f'{server_name}_server']

    if not server:
        pytest.skip(f'{server_name} server address not provided.')

    url = f"{server}/help"

    response = benchmark(requests.post, url)

    assert response.ok


class TestMySQL:
    @pytest.mark.parametrize('driver', DRIVERS)
    def test_ping(self, benchmark, benchmark_parameters, driver):
        mysql_server = benchmark_parameters['mysql_server']
        if not mysql_server:
            pytest.skip(f'MySQL server address not provided.')

        connection_uri = f"{MYSQL_USER}:{MYSQL_PWD}@{mysql_server}/{DATABASE_NAME}"
        engine = sqlalchemy.create_engine(f"{driver}://{connection_uri}")

        res = benchmark(engine.execute, "SELECT 1").fetchall()

        assert res

    @pytest.mark.parametrize('driver', DRIVERS)
    @pytest.mark.parametrize('query_name', QUERIES.keys())
    def test_query(self, benchmark, benchmark_parameters, driver, query_name):
        query = QUERIES[query_name]

        mysql_server = benchmark_parameters['mysql_server']
        if not mysql_server:
            pytest.skip(f'MySQL server address not provided.')

        connection_uri = f"{MYSQL_USER}:{MYSQL_PWD}@{mysql_server}/{DATABASE_NAME}"
        engine = sqlalchemy.create_engine(f"{driver}://{connection_uri}")

        res = benchmark(engine.execute, query).fetchall()

        assert res


class TestSearch:
    def test_text(self, benchmark, benchmark_parameters):
        search_server = benchmark_parameters['search_server']

        if not search_server:
            pytest.skip('Search server address not provided.')

        url = f"{search_server}/"
        payload_json = {'query_text': 'Glucose is a risk factor for COVID-19',
                        'k': 20,
                        'which_model': 'BSV'}

        response = benchmark(requests.post, url, json=payload_json)

        assert response.ok

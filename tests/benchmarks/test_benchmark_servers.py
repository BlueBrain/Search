import pytest
import requests
import sqlalchemy

MYSQL_USER = 'guest'
MYSQL_PWD = 'guest'
DATABASE_NAME = 'cord19_v35'


@pytest.mark.parametrize('server_name', ['embedding', 'mining', 'search'])
def test_flask_help(benchmark, benchmark_parameters, server_name):
    mining_server = benchmark_parameters[f'{server_name}_server']

    if not mining_server:
        pytest.skip(f'{server_name} server address not provided.')

    url = f"{mining_server}/help"

    response = benchmark(requests.post, url)

    assert response.ok


class TestMySQL:
    @pytest.mark.parametrize('driver', ['mysql+pymysql',
                                        'mysql+mysqldb',
                                        'mysql'])
    def test_ping(self, benchmark, benchmark_parameters, driver):
        mysql_server = benchmark_parameters['mysql_server']

        connection_uri = f"{MYSQL_USER}:{MYSQL_PWD}@{mysql_server}/{DATABASE_NAME}"
        engine = sqlalchemy.create_engine(f"{driver}://{connection_uri}")

        res = benchmark(engine.execute, "SELECT 1").fetchall()

        assert res

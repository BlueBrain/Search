import pandas as pd
import pytest
import requests
import sqlalchemy

# Embedding
EMBEDDING_MODELS = ["BSV", "SBioBERT", "SBERT", "USE"]

# Mining
ENTITY_TYPES = ["DISEASE", "CELL_TYPE", "CHEMICAL", "PROTEIN", "ORGAN"]
ARTICLE_IDS = [1234]

# MySQL
MYSQL_USER = "guest"
MYSQL_PWD = "guest"
DATABASE_NAME = "cord19_v35"
DRIVERS = ["mysql+pymysql", "mysql+mysqldb", "mysql"]
QUERIES = {"date_range": "SELECT article_id FROM articles WHERE "
                         "publish_time BETWEEN '1999-01-01' AND '2020-12-31'"}


@pytest.mark.parametrize("server_name", ["embedding", "mining", "search"])
def test_flask_help(benchmark, benchmark_parameters, server_name):
    """Ping the help route."""
    server = benchmark_parameters[f"{server_name}_server"]

    if not server:
        pytest.skip(f"{server_name} server address not provided.")

    url = f"{server}/help"

    response = benchmark(requests.post, url)

    assert response.ok


class TestEmbedding:
    @pytest.mark.parametrize("model", EMBEDDING_MODELS)
    def test_embed(self, benchmark, benchmark_parameters, model):
        """Embed a sentence with different models."""
        embedding_server = benchmark_parameters["embedding_server"]

        if not embedding_server:
            pytest.skip("Embedding server address not provided.")

        url = f"{embedding_server}/v1/embed/json"

        payload_json = {"text": "Glucose is a risk factor for COVID-19",
                        "model": model}

        response = benchmark(requests.post, url, json=payload_json)

        assert response.ok


class TestMining:
    @pytest.mark.parametrize("entity_type", ENTITY_TYPES)
    def test_mine_entity_text(self, benchmark, benchmark_parameters, entity_type):
        """Send 100 sentences as raw text."""
        mining_server = benchmark_parameters["mining_server"]

        if not mining_server:
            pytest.skip("Mining server address not provided.")

        url = f"{mining_server}/text"
        text = "Glucose is mainly made by plants during" \
               " photosynthesis from water and carbon dioxide."

        text *= 100

        header = ["entity_type", "property", "property_type", "property_value_type",
                  "ontology_source"]

        table = pd.Series({"entity_type": entity_type}, index=header).to_frame().transpose()
        schema_request = table.to_csv(index=False)

        payload_json = {"text": text, "schema": schema_request}

        response = benchmark(requests.post, url, json=payload_json)

        assert response.ok

    @pytest.mark.parametrize("entity_type", ENTITY_TYPES)
    @pytest.mark.parametrize("article_id", ARTICLE_IDS)
    def test_mine_entity_article(self, benchmark, benchmark_parameters, entity_type, article_id):
        """Mine an entire article from the database."""
        mining_server = benchmark_parameters["mining_server"]

        if not mining_server:
            pytest.skip("Mining server address not provided.")

        url = f"{mining_server}/database"
        identifiers = [(article_id, -1)]

        header = ["entity_type", "property", "property_type", "property_value_type",
                  "ontology_source"]

        table = pd.Series({"entity_type": entity_type}, index=header).to_frame().transpose()
        schema_request = table.to_csv(index=False)

        payload_json = {"identifiers": identifiers, "schema": schema_request}

        response = benchmark(requests.post, url, json=payload_json)

        assert response.ok


class TestMySQL:
    @pytest.mark.parametrize("driver", DRIVERS)
    def test_ping(self, benchmark, benchmark_parameters, driver):
        """Ping the database."""
        mysql_server = benchmark_parameters["mysql_server"]
        if not mysql_server:
            pytest.skip("MySQL server address not provided.")

        connection_uri = f"{MYSQL_USER}:{MYSQL_PWD}@{mysql_server}/{DATABASE_NAME}"
        engine = sqlalchemy.create_engine(f"{driver}://{connection_uri}")

        res = benchmark(engine.execute, "SELECT 1").fetchall()

        assert res

    @pytest.mark.parametrize("driver", DRIVERS)
    @pytest.mark.parametrize("query_name", QUERIES.keys())
    def test_query(self, benchmark, benchmark_parameters, driver, query_name):
        """Run predefined queries for different drivers."""
        query = QUERIES[query_name]

        mysql_server = benchmark_parameters["mysql_server"]
        if not mysql_server:
            pytest.skip("MySQL server address not provided.")

        connection_uri = f"{MYSQL_USER}:{MYSQL_PWD}@{mysql_server}/{DATABASE_NAME}"
        engine = sqlalchemy.create_engine(f"{driver}://{connection_uri}")

        res = benchmark(engine.execute, query).fetchall()

        assert res


class TestSearch:
    def test_text(self, benchmark, benchmark_parameters):
        """Search the most relevant sentences."""
        search_server = benchmark_parameters["search_server"]

        if not search_server:
            pytest.skip("Search server address not provided.")

        url = f"{search_server}/"
        payload_json = {"query_text": "Glucose is a risk factor for COVID-19",
                        "k": 20,
                        "which_model": "BSV"}

        response = benchmark(requests.post, url, json=payload_json)

        assert response.ok

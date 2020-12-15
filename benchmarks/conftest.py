"""Configuration of pytest benchmarks."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--embedding_server", default="", help="Embedding server URI")
    parser.addoption("--mining_server", default="", help="Mining server URI")
    parser.addoption("--mysql_server", default="", help="MySQL server URI")
    parser.addoption("--search_server", default="", help="Search server URI")


@pytest.fixture(scope="session")
def benchmark_parameters(request):
    return {
        "embedding_server": request.config.getoption("--embedding_server"),
        "mining_server": request.config.getoption("--mining_server"),
        "mysql_server": request.config.getoption("--mysql_server"),
        "search_server": request.config.getoption("--search_server"),
    }

import pytest
import subprocess


@pytest.mark.parametrize(
    "entrypoint_name",
    ["embedding_server", "create_database", "compute_embeddings", "search_server"]
)
def test_entrypoint(entrypoint_name):
    subprocess.check_call([entrypoint_name, "--help"])

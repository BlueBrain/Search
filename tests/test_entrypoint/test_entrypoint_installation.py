import subprocess

import pytest


@pytest.mark.parametrize(
    "entrypoint_name",
    [
        "compute_embeddings",
        "create_database",
        "create_mining_cache",
        "embedding_server",
        "mining_server",
        "search_server",
    ],
)
def test_entrypoint(entrypoint_name):
    subprocess.check_call([entrypoint_name, "--help"])

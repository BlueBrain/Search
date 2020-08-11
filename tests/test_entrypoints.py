import subprocess

import pytest


@pytest.mark.parametrize(
    "entrypoint_name",
    [
        "embedding_server",
        "create_database",
        "compute_embeddings",
        "search_server",
        "mining_server",
    ],
)
def test_entrypoint(entrypoint_name):
    subprocess.check_call([entrypoint_name, "--help"])

"""Subpackage containing all the entry points."""
from .compute_embeddings import run_compute_embeddings
from .create_database import run_create_database
from .create_mining_cache import run_create_mining_cache
from .embedding_server import get_embedding_app, run_embedding_server
from .mining_server import get_mining_app, run_mining_server
from .search_server import get_search_app, run_search_server

__all__ = [
    "get_embedding_app",
    "get_mining_app",
    "get_search_app",
    "run_compute_embeddings",
    "run_create_mining_cache",
    "run_create_database",
    "run_embedding_server",
    "run_mining_server",
    "run_search_server",
]

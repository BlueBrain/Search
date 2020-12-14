"""Collection of tests focusing on the `embedding_server` entrypoint."""
import pathlib
from unittest.mock import Mock

from bbsearch.entrypoint import get_embedding_app
from bbsearch.server.embedding_server import EmbeddingServer


def test_environment_reading(monkeypatch, tmpdir):
    tmpdir = pathlib.Path(str(tmpdir))
    logfile = tmpdir / "log.txt"
    logfile.touch()

    fake_embedding_server_inst = Mock(spec=EmbeddingServer)
    fake_embedding_server_class = Mock(return_value=fake_embedding_server_inst)

    monkeypatch.setattr(
        "bbsearch.server.embedding_server.EmbeddingServer", fake_embedding_server_class
    )

    # Mock all of our embedding models
    embedding_models = ["BSV", "SBioBERT", "Sent2VecModel", "SentTransformer", "USE"]

    for model in embedding_models:
        monkeypatch.setattr(f"bbsearch.embedding_models.{model}", Mock())

    monkeypatch.setenv("BBS_EMBEDDING_LOG_FILE", str(logfile))
    monkeypatch.setenv("BBS_EMBEDDING_BSV_CHECKPOINT_PATH", str(tmpdir / "fake_1.ckp"))
    monkeypatch.setenv(
        "BBS_EMBEDDING_SENT2VEC_CHECKPOINT_PATH", str(tmpdir / "fake_2.ckp")
    )

    embedding_app = get_embedding_app()

    assert embedding_app is fake_embedding_server_inst

    args, _ = fake_embedding_server_class.call_args

    assert len(args) == 1
    assert isinstance(args[0], dict)

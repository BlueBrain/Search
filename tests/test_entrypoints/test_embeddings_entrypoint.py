"""Collection of tests focusing on the compute_embeddings entrypoint."""
import pathlib
from unittest.mock import Mock

import numpy as np
import pytest

from bbsearch.entrypoints.embeddings_entrypoint import main


@pytest.mark.parametrize(
    (
        "batch_size_inference",
        "batch_size_transfer",
        "gpus",
        "custom_ixs",
        "model",
        "n_processes",
        "outfile",
        "temp_dir",
    ),
    [
        (11, 22, [0, 3], True, "BSV", 2, "some_out_dir/emb.h5", "some_temp_dir"),
        (1, 5, None, False, "Sent2Vec", 3, "new_out_dir/emb.h5", "new_temp_dir"),
        (3, 52, [1, None, 5, None], True, "SBioBERT", 4, "dir/emb.h5", "temp_dir"),
    ],
)
def test_sendthrough(
    monkeypatch,
    tmpdir,
    batch_size_inference,
    batch_size_transfer,
    gpus,
    custom_ixs,
    model,
    n_processes,
    outfile,
    temp_dir,
):
    """Check that CLI parameters correctly send through to __init__."""
    tmpdir = pathlib.Path(str(tmpdir))

    # Patching
    fake_mpe_class = Mock()
    fake_sqlalchemy = Mock()

    monkeypatch.setattr("bbsearch.embedding_models.MPEmbedder", fake_mpe_class)
    monkeypatch.setattr(
        "bbsearch.entrypoints.embeddings_entrypoint.sqlalchemy", fake_sqlalchemy
    )

    # Prepare CLI input
    args_and_opts = [
        model,
        outfile,
        f"--batch-size-inference={batch_size_inference}",
        f"--batch-size-transfer={batch_size_transfer}",
        f"--log-dir={str(tmpdir)}",
        f"--n-processes={n_processes}",
        f"--temp-dir={temp_dir}",
    ]

    if gpus is not None:
        gpus_option = ",".join([str(x) if x is not None else "" for x in gpus])
        args_and_opts.append(f"--gpus={gpus_option}")

    if custom_ixs:
        # We have a numpy array that defines them
        indices = np.array([2, 5, 8])
        indices_path = tmpdir / "indices.npy"
        np.save(indices_path, indices)
        args_and_opts.append(f"--indices-path={indices_path}")

    else:
        # We take all the sentences from our database (assuming there are 10)
        n_sentences = 10
        indices = np.arange(1, n_sentences + 1)
        fake_sqlalchemy.create_engine().execute.return_value = [(n_sentences,)]

    # Run CLI
    main(args_and_opts)

    # Checks
    if not custom_ixs:
        fake_sqlalchemy.create_engine().execute.assert_called_once()

    fake_mpe_class.assert_called_once()  # construction

    args, kwargs = fake_mpe_class.call_args

    assert args[0] == model
    assert args[1] is None
    np.testing.assert_array_equal(args[3], indices)
    assert args[4] == pathlib.Path(outfile)

    assert kwargs["batch_size_inference"] == batch_size_inference
    assert kwargs["batch_size_transfer"] == batch_size_transfer
    assert kwargs["n_processes"] == n_processes
    assert kwargs["gpus"] == gpus

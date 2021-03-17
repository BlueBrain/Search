"""Collection of tests focusing on the compute_embeddings entrypoint."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pathlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from bluesearch.entrypoint import run_compute_embeddings

N_GPUS = torch.cuda.device_count()


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
        (11, 22, [0, 3], True, "SBioBERT", 2, "some_out_dir/emb.h5", "some_temp_dir"),
        (1, 5, None, False, "SBioBERT", 3, "new_out_dir/emb.h5", "new_temp_dir"),
        (3, 52, [1, None, 5, None], True, "SBioBERT", 4, "dir/emb.h5", "temp_dir"),
    ],
)
def test_send_through(
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

    monkeypatch.setattr("bluesearch.embedding_models.MPEmbedder", fake_mpe_class)
    monkeypatch.setattr(
        "bluesearch.entrypoint.compute_embeddings.sqlalchemy", fake_sqlalchemy
    )

    # Prepare CLI input
    db_url = "my-sql-server.ch/my_db"
    args_and_opts = [
        model,
        outfile,
        f"--db-url={db_url}",
        f"--batch-size-inference={batch_size_inference}",
        f"--batch-size-transfer={batch_size_transfer}",
        f"--log-file={str(tmpdir / 'my.log')}",
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
    run_compute_embeddings(args_and_opts)

    # Checks
    if not custom_ixs:
        fake_sqlalchemy.create_engine().execute.assert_called_once()

    fake_mpe_class.assert_called_once()  # construction

    args, kwargs = fake_mpe_class.call_args

    assert args[1] == model
    np.testing.assert_array_equal(args[2], indices)
    assert args[3] == pathlib.Path(outfile)

    assert kwargs["batch_size_inference"] == batch_size_inference
    assert kwargs["batch_size_transfer"] == batch_size_transfer
    assert kwargs["n_processes"] == n_processes
    assert kwargs["gpus"] == gpus


@pytest.mark.slow
@pytest.mark.parametrize(
    "gpus",
    [
        pytest.param(",", id="CPU"),
        pytest.param(
            "0,1",
            id="GPU",
            marks=pytest.mark.skipif(N_GPUS < 2, reason="One needs at least 2 GPUs"),
        ),
    ],
)
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
@pytest.mark.parametrize("model", ["SBioBERT", "SBERT"])
def test_mp_real(
    tmpdir,
    monkeypatch,
    backend_database,
    fake_sqlalchemy_engine,
    model,
    start_method,
    gpus,
):
    tmpdir = pathlib.Path(str(tmpdir))
    fake_sqlalchemy = Mock()
    fake_sqlalchemy.create_engine.return_value = fake_sqlalchemy_engine
    monkeypatch.setattr(
        "bluesearch.entrypoint.compute_embeddings.sqlalchemy", fake_sqlalchemy
    )

    outfile = tmpdir / "output.h5"
    db_url = fake_sqlalchemy_engine.url
    n_processes = 2

    args_and_opts = [
        model,
        str(outfile),
        f"--db-url={db_url}",  # patched, however it is required for the CLI
        f"--gpus={gpus}",
        f"--log-file={str(tmpdir / 'my.log')}",
        f"--n-processes={n_processes}",
        f"--start-method={start_method}",
    ]

    assert not outfile.exists()

    run_compute_embeddings(args_and_opts)

    assert outfile.exists()

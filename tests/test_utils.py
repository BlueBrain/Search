"""Collection of tests focused on the utils.py module."""

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

import json
import pathlib

import h5py
import numpy as np
import pytest
import spacy

from bluesearch.utils import (
    H5,
    JSONL,
    Timer,
    check_entity_type_consistency,
    get_available_spacy_models,
    load_spacy_model,
)


class TestTimer:
    def test_errors(self):
        timer = Timer()

        with pytest.raises(ValueError):
            timer.__enter__()

        with pytest.raises(ValueError):
            with timer("a"):
                pass
            with timer("a"):
                pass

        with pytest.raises(ValueError):
            with timer("overall"):
                pass

        with pytest.raises(ValueError):
            with timer("b"):
                raise ValueError

        assert "b" not in timer.stats

    def test_basic(self):
        timer = Timer()

        assert len(timer.stats) == 1

        with timer("a"):
            pass

        assert "a" in timer.stats
        assert timer["a"] == timer.stats["a"]

        with timer("b"):
            pass

        assert len(timer.stats) == 3

    def test_verbose(self, capsys):
        timer_v = Timer(verbose=True)
        timer_n = Timer(verbose=False)

        # 1
        timer_v("b", message="additional")

        captured = capsys.readouterr()
        assert captured.out == "additional\n"

        # 2
        with timer_n("a"):
            pass

        captured = capsys.readouterr()
        assert not captured.out

        # 3
        with timer_v("a"):
            pass

        captured = capsys.readouterr()
        assert captured.out


class TestH5:
    @pytest.mark.parametrize("fillvalue", [1.1, np.nan])
    @pytest.mark.parametrize("indices", [[0, 1], [0], [3, 2, 1]])
    def test_clear(self, tmpdir, fillvalue, indices):
        h5_path = pathlib.Path(str(tmpdir)) / "to_be_created.h5"

        shape = (4, 2)
        data = np.random.random(shape)
        indices = np.array(indices)

        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset("a", shape=shape, fillvalue=fillvalue)

            dset[:, :] = data

        # Truth
        data[indices] = fillvalue

        # Run
        H5.clear(h5_path, "a", indices)

        with h5py.File(h5_path, "r") as ff:
            res = ff["a"][:]

        assert np.allclose(res, data, equal_nan=True)

    @pytest.mark.parametrize("batch_size", [1, 3, 10])
    @pytest.mark.parametrize("delete_inputs", [True, False])
    def test_concatenate(self, tmpdir, batch_size, delete_inputs):
        tmpdir = pathlib.Path(str(tmpdir))

        temp_path_1 = tmpdir / "temp_1.h5"
        temp_path_2 = tmpdir / "temp_2.h5"

        final_path = tmpdir / "final_1.h5"

        # Create temporary files
        indices_1 = [2, 5, 7]
        indices_2 = [1, 3, 8, 11]
        dim = 4
        shape_1 = (len(indices_1), dim)
        shape_2 = (len(indices_2), dim)

        array_1 = np.random.random(shape_1)
        array_2 = np.random.random(shape_2)
        dataset_name = "some_dataset"
        dataset_name_indices = f"{dataset_name}_indices"

        with h5py.File(temp_path_1, "w") as f_1:
            f_1.create_dataset(
                dataset_name,
                shape=shape_1,
            )
            f_1.create_dataset(
                dataset_name_indices,
                dtype="int32",
                shape=(len(indices_1), 1),
            )
            f_1[dataset_name][:] = array_1
            f_1[dataset_name_indices][:, 0] = np.array(indices_1, dtype=np.int32)

        with h5py.File(temp_path_2, "w") as f_2:
            f_2.create_dataset(
                dataset_name,
                shape=shape_2,
            )
            f_2.create_dataset(
                dataset_name_indices,
                dtype="int32",
                shape=(len(indices_2), 1),
            )
            f_2[dataset_name][:] = array_2
            f_2[dataset_name_indices][:, 0] = np.array(indices_2, dtype=np.int32)

        # No paths
        with pytest.raises(ValueError):
            H5.concatenate(final_path, dataset_name, [])

        # Overlapping indices
        with pytest.raises(ValueError):
            H5.concatenate(final_path, dataset_name, [temp_path_1, temp_path_1])

        H5.concatenate(
            final_path,
            dataset_name,
            [temp_path_1, temp_path_2],
            delete_inputs=delete_inputs,
            batch_size=batch_size,
        )

        if delete_inputs:
            assert not temp_path_1.exists()
            assert not temp_path_2.exists()
        else:
            assert temp_path_1.exists()
            assert temp_path_2.exists()

        assert final_path.exists()

        with h5py.File(final_path, "r") as f:
            data = f[dataset_name][:]

        assert data.shape == (max(indices_1 + indices_2) + 1, dim)
        np.testing.assert_array_almost_equal(data[indices_1], array_1)
        np.testing.assert_array_almost_equal(data[indices_2], array_2)

        assert np.all(
            H5.find_populated_rows(final_path, dataset_name)
            == np.array(sorted(indices_1 + indices_2))
        )

    def test_create(self, tmpdir):
        h5_path = pathlib.Path(str(tmpdir)) / "to_be_created.h5"

        # New h5 file and new dataset
        H5.create(h5_path, "a", (20, 10), dtype="f2")

        with h5py.File(h5_path, "r") as f:
            assert "a" in f.keys()
            assert f["a"].shape == (20, 10)
            assert f["a"].dtype == "f2"

        # Old h5 file and new dataset
        H5.create(h5_path, "b", (40, 2), dtype="f4")

        with h5py.File(h5_path, "r") as f:
            assert "b" in f.keys()
            assert f["b"].shape == (40, 2)
            assert f["b"].dtype == "f4"

        # Check errors
        with pytest.raises(ValueError):
            H5.create(h5_path, "a", (20, 10))

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("model", ["SBERT"])
    def test_find_unpopulated_rows(
        self, embeddings_h5_path, model, verbose, batch_size
    ):
        unpop_rows_computed = H5.find_unpopulated_rows(
            embeddings_h5_path, model, verbose=verbose, batch_size=batch_size
        )

        with h5py.File(embeddings_h5_path, "r") as f:
            dset_np = f[model][:]
            unpop_rows_true = np.where(np.isnan(dset_np.sum(axis=1)))[0]

        assert np.all(unpop_rows_computed == unpop_rows_true)

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("model", ["SBERT"])
    def test_find_populated_rows(self, embeddings_h5_path, model, verbose, batch_size):
        unpop_rows_computed = H5.find_populated_rows(
            embeddings_h5_path, model, verbose=verbose, batch_size=batch_size
        )

        with h5py.File(embeddings_h5_path, "r") as f:
            dset_np = f[model][:]
            unpop_rows_true = np.where(~np.isnan(dset_np.sum(axis=1)))[0]

        assert np.all(unpop_rows_computed == unpop_rows_true)

    def test_get_shape(self, tmpdir):
        h5_path = pathlib.Path(str(tmpdir)) / "to_be_created.h5"

        shape = (22, 3)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("a", shape=shape)

        assert H5.get_shape(h5_path, "a") == shape

    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    @pytest.mark.parametrize("model", ["SBERT"])
    @pytest.mark.parametrize(
        "indices",
        [
            [10, 1, 0, 4, 6, 2, 12, 5],
            [1],
            [1, 2],
            [6, 5, 4, 3, 2, 1, 0],
            [1, 5, 2, 6, 11, 12, 14],
        ],
    )
    def test_load(self, embeddings_h5_path, model, verbose, batch_size, indices):
        with h5py.File(embeddings_h5_path, "r") as f:
            dset_np = f[model][:]

        res_loaded = H5.load(
            embeddings_h5_path,
            model,
            indices=np.array(indices),
            verbose=verbose,
            batch_size=batch_size,
        )

        res_true = dset_np[indices]

        assert isinstance(res_loaded, np.ndarray)
        assert res_loaded.shape == res_true.shape

        # Check nans are identical
        assert np.all(np.isnan(res_loaded) == np.isnan(res_true))

        # Check nonnan entries
        nonnan_mask = ~np.isnan(res_loaded)
        assert np.allclose(res_loaded[nonnan_mask], res_true[nonnan_mask])

    def test_load_duplicates(self, embeddings_h5_path):
        with pytest.raises(ValueError):
            H5.load(embeddings_h5_path, "SBERT", indices=np.array([1, 2, 2]))

    @pytest.mark.parametrize("flip", [True, False])
    def test_write(self, tmpdir, flip):
        h5_path = pathlib.Path(str(tmpdir)) / "to_be_created.h5"

        shape = (20, 3)
        dtype_h5 = "f4"
        dtype_np = "float32"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("a", shape=shape, dtype=dtype_h5, fillvalue=np.nan)

        data = np.random.random((10, 3)).astype(dtype_np)
        indices = np.arange(0, 20, 2)
        if flip:
            indices = np.flip(indices)

        indices_complement = np.setdiff1d(np.arange(shape[0]), indices)

        H5.write(h5_path, "a", data, indices)

        with h5py.File(h5_path, "r") as f:
            res_np = f["a"][:]

        assert res_np.shape == shape
        assert np.allclose(res_np[indices], data)
        assert np.all(np.isnan(res_np[indices_complement]))


def test_load_save_jsonl(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "file.jsonl"

    li = [{"a": 1, "b": "cc"}, {"k": 23}]
    JSONL.dump_jsonl(li, path)
    lo = JSONL.load_jsonl(path)

    assert li == lo


@pytest.mark.parametrize(
    "metadata,expected_value",
    [
        ({"wrongkey": "wrongvalue"}, False),  # labels key missing
        ({"labels": "wrongvalue"}, False),  # ner key missing
        ({"labels": {"ner": ["SEVERAL", "ENTITIES"]}}, False),  # several entities
        ({"labels": {"ner": ["organism"]}}, False),  # entity lower
        ({"labels": {"ner": ["ORGANISM"]}}, True),  # Good structure
    ],
)
def test_check_entity_type_consistency(tmpdir, metadata, expected_value):
    """Test that checks are working"""
    # Wrong model directory name
    wrong_model_dir = pathlib.Path(tmpdir) / "models" / "ner_er" / "wrongmodelname"
    wrong_model_dir.mkdir(parents=True)
    assert check_entity_type_consistency(wrong_model_dir) is False

    # Missing meta.json file
    good_model_dir = pathlib.Path(tmpdir) / "models" / "ner_er" / "model-organism"
    good_model_dir.mkdir(parents=True)
    assert check_entity_type_consistency(good_model_dir) is False

    # meta.json structure
    meta_path = good_model_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    assert check_entity_type_consistency(good_model_dir) is expected_value


def test_get_available_spacy_models(spacy_model_path, entity_types):
    """Test that available spacy models."""
    models_dir = spacy_model_path / "models" / "ner_er"
    expected_results = {
        etype.upper(): models_dir / f"model-{etype.lower()}" for etype in entity_types
    }
    results = get_available_spacy_models(spacy_model_path)
    assert isinstance(results, dict)
    assert len(results) == 2
    assert sorted(results.keys()) == sorted(entity_types)
    assert results == expected_results


@pytest.mark.filterwarnings()
def test_get_available_spacy_models_warning(tmpdir):
    """Test that available spacy models warnings."""
    wrong_path = pathlib.Path(tmpdir) / "wrongpath"
    wrong_model = wrong_path / "models" / "ner_er" / "wrongmodel"
    wrong_model.mkdir(parents=True)
    with pytest.warns(UserWarning):
        _ = get_available_spacy_models(wrong_path)


@pytest.mark.parametrize(
    "model_name,is_found", [("en_core_web_sm", True), ("xx_xxxx_xxx_xx", False)]
)
def test_load_spacy_model(model_name, is_found):
    if is_found:
        nlp = load_spacy_model(model_name)
        assert isinstance(nlp, spacy.language.Language)
    else:
        with pytest.raises(ModuleNotFoundError):
            load_spacy_model(model_name)

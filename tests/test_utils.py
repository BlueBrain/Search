"""Collection of tests focused on the utils.py module."""
import pathlib

import h5py
import numpy as np
import pytest

from bbsearch.utils import Timer, H5


class TestTimer:

    def test_errors(self):
        timer = Timer()

        with pytest.raises(ValueError):
            timer.__enter__()

        with pytest.raises(ValueError):
            with timer('a'):
                pass
            with timer('a'):
                pass

        with pytest.raises(ValueError):
            with timer('overall'):
                pass

        with pytest.raises(ValueError):
            with timer('b'):
                raise ValueError

        assert 'b' not in timer.stats

    def test_basic(self):
        timer = Timer()

        assert len(timer.stats) == 1

        with timer('a'):
            pass

        assert 'a' in timer.stats
        assert timer['a'] == timer.stats['a']

        with timer('b'):
            pass

        assert len(timer.stats) == 3

    def test_verbose(self, capsys):
        timer_v = Timer(verbose=True)
        timer_n = Timer(verbose=False)

        # 1
        timer_v('b', message='additional')

        captured = capsys.readouterr()
        assert captured.out == 'additional\n'

        # 2
        with timer_n('a'):
            pass

        captured = capsys.readouterr()
        assert not captured.out

        # 3
        with timer_v('a'):
            pass

        captured = capsys.readouterr()
        assert captured.out


class TestH5:

    def test_create(self, tmpdir):
        h5_path = pathlib.Path(str(tmpdir)) / 'to_be_created.h5'

        # New h5 file and new dataset
        H5.create(h5_path, 'a', (20, 10), dtype='f2')

        with h5py.File(h5_path, 'r') as f:
            assert 'a' in f.keys()
            assert f['a'].shape == (20, 10)
            assert f['a'].dtype == 'f2'

        # Old h5 file and new dataset
        H5.create(h5_path, 'b', (40, 2), dtype='f4')

        with h5py.File(h5_path, 'r') as f:
            assert 'b' in f.keys()
            assert f['b'].shape == (40, 2)
            assert f['b'].dtype == 'f4'

        # Check errors
        with pytest.raises(ValueError):
            H5.create(h5_path, 'a', (20, 10))

    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    @pytest.mark.parametrize('model', ['SBERT'])
    def test_find_unpopulated_rows(self, embeddings_h5_path, model, verbose, batch_size):
        unpop_rows_computed = H5.find_unpopulated_rows(embeddings_h5_path,
                                                       model,
                                                       verbose=verbose,
                                                       batch_size=batch_size)

        with h5py.File(embeddings_h5_path, 'r') as f:
            dset_np = f[model][:]
            unpop_rows_true = np.where(np.isnan(dset_np.sum(axis=1)))[0]

        assert np.all(unpop_rows_computed == unpop_rows_true)

    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    @pytest.mark.parametrize('model', ['SBERT'])
    def test_find_populated_rows(self, embeddings_h5_path, model, verbose, batch_size):
        unpop_rows_computed = H5.find_populated_rows(embeddings_h5_path,
                                                     model,
                                                     verbose=verbose,
                                                     batch_size=batch_size)

        with h5py.File(embeddings_h5_path, 'r') as f:
            dset_np = f[model][:]
            unpop_rows_true = np.where(~np.isnan(dset_np.sum(axis=1)))[0]

        assert np.all(unpop_rows_computed == unpop_rows_true)

    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    @pytest.mark.parametrize('model', ['SBERT'])
    @pytest.mark.parametrize('indices', [
        [10, 1, 0, 4, 6, 2, 12, 5],
        [1],
        [1, 2],
        [6, 5, 4, 3, 2, 1, 0],
        [1, 5, 2, 6, 11, 12, 14]])
    def test_load(self, embeddings_h5_path, model, verbose, batch_size, indices):
        with h5py.File(embeddings_h5_path, 'r') as f:
            dset_np = f[model][:]

        res_loaded = H5.load(embeddings_h5_path,
                             model,
                             indices=np.array(indices),
                             verbose=verbose,
                             batch_size=batch_size)

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
            H5.load(embeddings_h5_path, 'SBERT', indices=np.array([1, 2, 2]))

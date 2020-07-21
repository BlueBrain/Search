"""Collection of tests focused on the utils.py module."""
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
    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    @pytest.mark.parametrize('model', ['SBERT', 'SBioBERT', 'USE', 'BSV'])
    def test_find_unpopulated_rows(self, h5_path, model, verbose, batch_size):
        unpop_rows_computed = H5.find_unpopulated_rows(h5_path,
                                                       model,
                                                       verbose=verbose,
                                                       batch_size=batch_size)

        with h5py.File(h5_path, 'r') as f:
            dset_np = f[model][:]
            unpop_rows_true = np.where(np.isnan(dset_np.sum(axis=1)))[0]

        assert np.all(unpop_rows_computed == unpop_rows_true)

    @pytest.mark.parametrize('verbose', [True, False])
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    @pytest.mark.parametrize('model', ['SBERT', 'SBioBERT', 'USE', 'BSV'])
    def test_load(self, h5_path, model, verbose, batch_size):
        with h5py.File(h5_path, 'r') as f:
            # n_sentences = len(f[model])
            dset_np = f[model][:]

        indices = np.array([10, 1, 0, 4, 6, 2, 12, 5])

        res_loaded = H5.load(h5_path,
                             model,
                             indices=indices,
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

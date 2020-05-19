"""Collection of tests focused on the utils.py module."""
import pytest

from bbsearch.utils import Timer


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

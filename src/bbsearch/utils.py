"""

Generic Utils waiting for migration in proper submodule.

"""
import time


class Timer:
    r"""Convenience context manager timing functions and logging the results.

    The order of execution is `__call__`,  `__enter__` and `__exit__`.

    Parameters
    ----------
    verbose : bool
        If True, whenever process ends we print the elapsed time to standard output.

    Attributes
    ----------
    inst_time : float
        Time of instantiation.

    name : str or None
        Name of the process to be timed. The user can control the value via the `__call__` magic.

    logs : dict
        Internal dictionary that stores all the times. The keys are the process names and the values are number
        of seconds.

    start_time : float or None
        Time of the last enter. Is dynamically changed when entering.

    Examples
    --------
    >>> import time
    >>> from bbsearch.utils import Timer
    >>>
    >>> timer = Timer(verbose=False)
    >>>
    >>> with timer('experiment_1'):
    ...     time.sleep(0.05)
    >>>
    >>> with timer('experiment_2'):
    ...     time.sleep(0.02)
    >>>
    >>> assert set(timer.stats.keys()) == {'overall', 'experiment_1', 'experiment_2'}
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

        self.inst_time = time.time()
        self.name = None  # what key is being populated
        self.logs = {}
        self.start_time = None  # to be overwritten when entering

    def __call__(self, name, message=None):
        """Define the name of the process to be timed.

        Parameters
        ----------
        name : str
            Name of the process to be timed.

        message : str or None
            Optional message to be printed to stoud when entering. Note that it only has an effect if
            `self.verbose=True`.

        """
        self.name = name

        if self.verbose and message is not None:
            print(message)

        return self

    def __enter__(self):
        """Launch the timer."""
        if self.name is None:
            raise ValueError('No name specified, one needs to call the instance with some name.')

        if self.name in self.logs:
            raise ValueError('{} has already been timed'.format(self.name))

        if self.name == 'overall':
            raise ValueError("The 'overall' key is restricted for length of the lifetime of the Timer.")

        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log internally."""
        if exc_type is not None:
            # raised an exception
            self.start_time = None
            self.name = None
            return False

        else:
            # nothing bad happened
            end_time = time.time()
            self.logs[self.name] = end_time - self.start_time

            if self.verbose:
                fmt = '{:.2f}'
                print("{} took ".format(self.name) + fmt.format(self.logs[self.name]) + ' seconds')

        # cleanup
        self.start_time = None
        self.name = None

    def __getitem__(self, item):
        """Get a single experiment."""
        return self.logs[item]

    @property
    def stats(self):
        """Return all timing statistics."""
        return {'overall': time.time() - self.inst_time, **self.logs}



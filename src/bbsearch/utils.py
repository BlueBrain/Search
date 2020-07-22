"""Generic Utils."""
import h5py
import numpy as np
import time
import tqdm


class Timer:
    """Convenience context manager timing functions and logging the results.

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
        Name of the process to be timed.
        The user can control the value via the `__call__` magic.

    logs : dict
        Internal dictionary that stores all the times.
        The keys are the process names and the values are number of seconds.

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

        self.inst_time = time.perf_counter()
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

        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log internally."""
        if exc_type is not None:
            # raised an exception
            self.start_time = None
            self.name = None
            return False

        else:
            # nothing bad happened
            end_time = time.perf_counter()
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
        return {'overall': time.perf_counter() - self.inst_time, **self.logs}


class H5:

    @staticmethod
    def clear(h5_path, dataset_name, indices):
        """Set selected rows to the fillvalue.

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        indices : np.ndarray
            1D array that determines the rows to be set to fillvalue.
        """
        with h5py.File(h5_path, 'a') as f:
            h5_dset = f[dataset_name]
            fillvalue = h5_dset.fillvalue
            dim = h5_dset.shape[1]

            argsort = indices.argsort()
            h5_dset[indices[argsort]] = np.ones((len(argsort), dim)) * fillvalue

    @staticmethod
    def create(h5_path, dataset_name, shape, dtype='f4', fillvalue=np.nan):
        """Create a dataset (and potentially also a h5 file).

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        shape : tuple
            Two element tuple representing rows and columns.

        dtype : str
            Dtype of the h5 array. See references for all the details.

        fillvalue : float
            How to fill unpopulated rows.

        References
        ----------
        [1] http://docs.h5py.org/en/stable/faq.html#faq
        """

        if h5_path.is_file():
            with h5py.File(h5_path, 'a') as f:
                if dataset_name in f.keys():
                    raise ValueError('The {} dataset already exists.'.format(dataset_name))

                f.create_dataset(dataset_name, shape=shape, dtype=dtype, fillvalue=fillvalue)

        else:
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset(dataset_name, shape=shape, dtype=dtype, fillvalue=fillvalue)

    @staticmethod
    def delete():
        pass

    @staticmethod
    def find_unpopulated_rows(h5_path, dataset_name, batch_size=100, verbose=False):
        """Identify rows that are unpopulated (= nan vectors).

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        batch_size : int
            Number of rows to be loaded at a time.

        verbose : bool
            Controls verbosity.

        Returns
        -------
        unpop_rows : np.ndarray
            1D numpy array of ints representing row indices of unpopulated rows (nan).

        """

        with h5py.File(h5_path, 'r') as f:
            dset = f[dataset_name]
            n_rows = len(dset)

            unpop_rows = []
            iterable = range(0, n_rows, batch_size)

            if verbose:
                iterable = tqdm.tqdm(iterable)

            for i in iterable:
                row = dset[i: i + batch_size]
                is_unpop = np.isnan(row.sum(axis=1))  # (batch_size,)

                unpop_rows.extend(list(np.where(is_unpop)[0] + i))

        return np.array(unpop_rows)

    @staticmethod
    def find_populated_rows(h5_path, dataset_name, batch_size=100, verbose=False):
        """Identify rows that are populated (= not nan vectors).

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        batch_size : int
            Number of rows to be loaded at a time.

        verbose : bool
            Controls verbosity.

        Returns
        -------
        pop_rows : np.ndarray
            1D numpy array of ints representing row indices of populated rows (not nan).

        """
        with h5py.File(h5_path, 'r') as f:
            dset = f[dataset_name]
            n_rows = len(dset)  # 7

        unpop_rows = H5.find_unpopulated_rows(h5_path,
                                              dataset_name,
                                              batch_size=batch_size,
                                              verbose=verbose)  # [2, 3, 6]

        pop_rows = np.setdiff1d(np.arange(n_rows), unpop_rows)  # [0, 1, 4, 5]

        return pop_rows

    @staticmethod
    def get_shape(h5_path, dataset_name):
        """Get shape of a dataset.

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.
        """

        with h5py.File(h5_path, 'r') as f:
            shape = f[dataset_name].shape

        return shape

    @staticmethod
    def load(h5_path, dataset_name, batch_size=500, indices=None, verbose=False):
        """Load an h5 file in memory.

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        batch_size : int
            Number of rows to be loaded at a time.

        indices : None or np.ndarray
            If None then we load all the rows from the dataset. If ``np.ndarray``
            then the loading only selected indices.

        verbose : bool
            Controls verbosity.

        Returns
        -------
        res : np.ndarray
            Numpy array of shape `(len(indices), ...)` holding the loaded rows.

        """
        with h5py.File(h5_path, 'r') as f:
            dset = f[dataset_name]
            n_rows = len(dset)

            indices = indices if indices is not None else np.arange(n_rows)  # [10, 9, 12, 1]

            if len(set(indices)) != len(indices):
                raise ValueError('There cannot be duplicates inside of the indices')

            argsort = indices.argsort()  # [3, 1, 0, 2]

            sorted_indices = indices[argsort]  # [1, 9, 10, 12]
            unargsort = argsort.argsort()  # [2, 1, 3, 0]

            final_res_l = []

            n_indices = len(sorted_indices)
            iterable = range(0, n_indices, batch_size)

            if verbose:
                iterable = tqdm.tqdm(iterable)

            for i in iterable:
                subarray = dset[sorted_indices[i: i + batch_size]]  # (batch_size, dim)
                final_res_l.append(subarray)

            final_res = np.concatenate(final_res_l, axis=0)

            return final_res[unargsort]

    @staticmethod
    def write(h5_path, dataset_name, data, indices):
        """Write a numpy array into an h5 file.

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.

        dataset_name : str
            Name of the dataset.

        data : np.ndarray
            2D numpy array to be written into the h5 file.

        indices : np.ndarray
            1D numpy array that determines row indices whre the `data` pasted.

        """

        with h5py.File(h5_path, 'a') as f:
            h5_dset = f[dataset_name]

            argsort = indices.argsort()
            h5_dset[indices[argsort]] = data[argsort]

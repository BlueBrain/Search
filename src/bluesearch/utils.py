"""Generic Utils."""

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
import time
import warnings
from typing import Any, Dict, Set, Union

import h5py
import numpy as np
import spacy


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
    >>> from bluesearch.utils import Timer
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
        self.start_time = float("nan")  # to be overwritten when entering

    def __call__(self, name, message=None):
        """Define the name of the process to be timed.

        Parameters
        ----------
        name : str
            Name of the process to be timed.
        message : str or None
            Optional message to be printed to stoud when entering. Note that
            it only has an effect if `self.verbose=True`.
        """
        self.name = name

        if self.verbose and message is not None:
            print(message)

        return self

    def __enter__(self):
        """Launch the timer."""
        if self.name is None:
            raise ValueError(
                "No name specified, one needs to call the instance with some name."
            )

        if self.name in self.logs:
            raise ValueError("{} has already been timed".format(self.name))

        if self.name == "overall":
            raise ValueError(
                "The 'overall' key is restricted for length of the "
                "lifetime of the Timer."
            )

        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log internally."""
        if exc_type is None:
            # nothing bad happened
            end_time = time.perf_counter()
            self.logs[self.name] = end_time - self.start_time

            if self.verbose:
                fmt = "{:.2f}"
                print(
                    "{} took ".format(self.name)
                    + fmt.format(self.logs[self.name])
                    + " seconds"
                )
        else:
            # an exception was raised in the context manager; clean up.
            self.start_time = float("nan")
            self.name = None

    def __getitem__(self, item):
        """Get a single experiment."""
        return self.logs[item]

    @property
    def stats(self):
        """Return all timing statistics."""
        return {"overall": time.perf_counter() - self.inst_time, **self.logs}


class H5:
    """H5 utilities."""

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
        with h5py.File(h5_path, "a") as f:
            h5_dset = f[dataset_name]
            fillvalue = h5_dset.fillvalue
            dim = h5_dset.shape[1]

            h5_dset[np.sort(indices)] = np.ones((len(indices), dim)) * fillvalue

    @staticmethod
    def concatenate(
        h5_path_output, dataset_name, h5_paths_temp, delete_inputs=True, batch_size=2000
    ):
        """Concatenate multiple h5 files into one h5 file.

        Parameters
        ----------
        h5_path_output : pathlib.Path
            Path to the h5 file. Note that this file can already exist and contain other
            datasets.
        dataset_name : str
            Name of the dataset.
        h5_paths_temp : list
            Paths to the input h5 files. Note that each of them will have 2 datasets.
                - `{dataset_name}` - dtype = float and shape (length, dim)
                - `{dataset_name}_indices` - dtype = int and shape (length, 1)
        delete_inputs : bool
            If True, then all input h5 files are deleted once the concatenation is done.
        batch_size : int
            Batch size to be used for transfers from the input h5 to the final one.
        """
        if not h5_paths_temp:
            raise ValueError("No temporary h5 files provided.")

        all_indices: Set[int] = set()
        dim = None
        for path_temp in h5_paths_temp:
            with h5py.File(path_temp, "r") as f:
                current_indices_set: Set[int] = set(f[f"{dataset_name}_indices"][:, 0])
                current_dim = f[f"{dataset_name}"].shape[1]

                if dim is None:
                    dim = current_dim
                else:
                    if current_dim != dim:
                        raise ValueError(
                            f"The dimension of {path_temp} is inconsistent"
                        )

                if all_indices & current_indices_set:
                    inters = all_indices & current_indices_set
                    raise ValueError(
                        f"{path_temp} introduces an overlapping index: {inters}"
                    )

                all_indices |= current_indices_set

        final_length = max(all_indices) + 1
        H5.create(h5_path_output, dataset_name, shape=(final_length, dim))

        for path_temp in h5_paths_temp:
            with h5py.File(path_temp, "r") as f:
                current_indices = f[f"{dataset_name}_indices"][:, 0]
                n_current_indices = len(current_indices)
                batch_size = min(n_current_indices, batch_size)

                batches = np.array_split(
                    np.arange(n_current_indices), n_current_indices / batch_size
                )
                h5_data = f[f"{dataset_name}"]
                for batch in batches:
                    H5.write(
                        h5_path_output,
                        dataset_name,
                        h5_data[batch],
                        current_indices[batch],
                    )

        if delete_inputs:
            for path_temp in h5_paths_temp:
                path_temp.unlink()

    @staticmethod
    def create(h5_path, dataset_name, shape, dtype="f4"):
        """Create a dataset (and potentially also a h5 file).

        Parameters
        ----------
        h5_path : pathlib.Path
            Path to the h5 file.
        dataset_name : str
            Name of the dataset.
        shape : tuple of int
            Two element tuple representing rows and columns.
        dtype : str
            Dtype of the h5 array. See references for all the details.

        Notes
        -----
        Unpopulated rows will be filled with `np.nan`.

        References
        ----------
        [1] http://docs.h5py.org/en/stable/faq.html#faq
        """
        if h5_path.is_file():
            with h5py.File(h5_path, "a") as f:
                if dataset_name in f.keys():
                    raise ValueError(
                        "The {} dataset already exists.".format(dataset_name)
                    )

                f.create_dataset(
                    dataset_name, shape=shape, dtype=dtype, fillvalue=np.nan
                )

        else:
            with h5py.File(h5_path, "w") as f:
                f.create_dataset(
                    dataset_name, shape=shape, dtype=dtype, fillvalue=np.nan
                )

    @staticmethod
    def find_unpopulated_rows(h5_path, dataset_name, batch_size=2000, verbose=False):
        """Return the indices of rows that are unpopulated.

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
        with h5py.File(h5_path, "r") as f:
            dset = f[dataset_name]
            n_rows = len(dset)
            unpop_rows = []

            for i in range(0, n_rows, batch_size):
                if verbose:
                    print(
                        f"\rFinding unpopulated rows: {round(100*i/n_rows):>3d}% done",
                        end="",
                    )
                row = dset[i : i + batch_size]
                is_unpop = np.isnan(row).any(axis=1)  # (batch_size,)
                unpop_rows.extend(list(np.where(is_unpop)[0] + i))

            print("\rFinding unpopulated rows: 100% done", end="")

        return np.array(unpop_rows)

    @staticmethod
    def find_populated_rows(h5_path, dataset_name, batch_size=2000, verbose=False):
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
        with h5py.File(h5_path, "r") as f:
            dset = f[dataset_name]
            n_rows = len(dset)  # 7

        unpop_rows = H5.find_unpopulated_rows(
            h5_path, dataset_name, batch_size=batch_size, verbose=verbose
        )  # [2, 3, 6]

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
        with h5py.File(h5_path, "r") as f:
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
        with h5py.File(h5_path, "r") as f:
            dset = f[dataset_name]

            if indices is None:
                return dset[:]

            if len(set(indices)) != len(indices):
                raise ValueError("There cannot be duplicates inside of the indices")

            argsort = indices.argsort()  # [3, 1, 0, 2]

            sorted_indices = indices[argsort]  # [1, 9, 10, 12]
            unargsort = np.empty_like(argsort)
            unargsort[argsort] = np.arange(len(argsort))  # [2, 1, 3, 0]

            final_res_l = []

            n_indices = len(sorted_indices)

            for i in range(0, n_indices, batch_size):
                if verbose:
                    print(f"\rLoading H5: {round(100*i/n_indices):>3d}% done", end="")
                subarray = dset[sorted_indices[i : i + batch_size]]  # (batch_size, dim)
                final_res_l.append(subarray)

            final_res = np.concatenate(final_res_l, axis=0)
            print("\rLoading H5: 100% done", end="")
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
        with h5py.File(h5_path, "a") as f:
            h5_dset = f[dataset_name]

            argsort = indices.argsort()
            h5_dset[indices[argsort]] = data[argsort]


class JSONL:
    """Collection of utility static functions handling `jsonl` files."""

    @staticmethod
    def dump_jsonl(data, path):
        """Save a list of dictionaries to a jsonl.

        Parameters
        ----------
        data : list
            List of dictionaries (json files).
        path : pathlib.Path
            File where to save it.
        """
        with path.open("w") as f:
            for x in data:
                line = json.dumps(x)
                f.write(line + "\n")

    @staticmethod
    def load_jsonl(path):
        """Read jsonl into a list of dictionaries.

        Parameters
        ----------
        path : pathlib.Path
            Path to the .jsonl file.

        Returns
        -------
        data : list
            List of dictionaries.
        """
        with path.open() as f:
            text = f.read()
            data = [json.loads(jline) for jline in text.splitlines()]

        return data


class MissingEnvironmentVariable(Exception):
    """Exception for missing environment variables."""


def check_entity_type_consistency(model_path: Union[str, pathlib.Path]) -> bool:
    """Check that entity type of the model name is the same as in the ner pipe.

    Parameters
    ----------
    model_path
        Path to a spacy model directory.

    Returns
    -------
    bool
        If true, the name of the model and the entity type name detected by the model
        are consistent. Otherwise, it is not.
    """
    model_path = pathlib.Path(model_path)

    _, dash, entity_type = model_path.stem.partition("-")

    if dash != "-" or not entity_type.islower():
        return False

    meta_file = model_path / "meta.json"

    if not meta_file.exists():
        return False

    with open(meta_file) as f:
        metadata = json.load(f)

    if "labels" not in metadata:
        return False
    if "ner" not in metadata["labels"]:
        return False

    detected_labels = metadata["labels"]["ner"]

    if len(detected_labels) != 1:
        return False

    detected_entity_type = detected_labels[0]

    if not detected_entity_type.isupper():
        return False

    return entity_type.upper() == detected_entity_type


def get_available_spacy_models(
    data_and_models_dir: Union[str, pathlib.Path]
) -> Dict[str, pathlib.Path]:
    """List available spacy models for a given data directory.

    Parameters
    ----------
    data_and_models_dir
        Path to directory "data_and_models".
        Should contains models/ner_er and models/er directories with all spacy models.

    Returns
    -------
    models_dict
        Dictionary mapping the entity type to the spacy model path detecting it.
        Only the models following the naming convention are kept.
    """
    data_and_models_dir = pathlib.Path(data_and_models_dir)

    models_dir = data_and_models_dir / "models" / "ner_er"

    available_models = [
        model_path for model_path in models_dir.iterdir() if model_path.is_dir()
    ]
    models_dict = {}
    for model_path in available_models:
        if not check_entity_type_consistency(model_path):
            warnings.warn(
                f"Name of the model {model_path} is not consistent with "
                "the detected entities. Therefore, this model was not "
                "included into the list of available models."
            )
        else:
            _, _, entity_type = model_path.stem.partition("-")
            models_dict[entity_type.upper()] = model_path.resolve()

    return models_dict


def load_spacy_model(
    model_name: Union[str, pathlib.Path], *args: Any, **kwargs: Any
) -> spacy.language.Language:
    """Spacy model load with informative error message.

    Parameters
    ----------
    model_name:
        spaCy pipeline to load. It can be a package name or a local path.
    *args, **kwargs:
        Arguments passed to `spacy.load()`

    Returns
    -------
    model:
        Loaded spaCy pipeline.

    Raises
    ------
    ModuleNotFoundError
        If spaCy model loading failed due to non-existent package or local file.
    """
    try:
        return spacy.load(model_name, *args, **kwargs)
    except IOError as err:
        if str(err).startswith("[E050]"):
            raise ModuleNotFoundError(
                "Failed to load the following spaCy model:"
                f'    model_name = "{model_name}"'
                "If model_name is a package name, please install it using"
                "    $ pip install ..."
                "If model_name is a local path, please verify the pipeline path."
            ) from err
        else:
            raise

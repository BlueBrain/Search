"""Module for handling identifiers."""

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

import functools
from typing import List, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray


def generate_uuids(metadata: pd.DataFrame, identifiers: List[str]) -> pd.DataFrame:
    """Generate UUIDs for papers with multiple sources / identifiers.

    Papers are clustered if they have the same or no value for the given identifiers.
    Each paper cluster is assigned a UUID.

    Parameters
    ----------
    metadata
        Paper metadata including the given identifiers.
    identifiers
        Columns of the identifiers.

    Returns
    -------
    pandas.DataFrame
        Generated UUIDs with the given identifiers. The index is the same as the input.
    """

    def step_columns(step: int) -> List[str]:
        """Create column names for the given step and identifiers."""
        return [f"step{step}_{x}" for x in identifiers]

    # Step 0: Create a copy of the metadata with only the identifiers.

    df = metadata[identifiers].copy()

    # Step 1: Cluster papers per identifier, column-wise.

    # The types of the output list elements are a bit too complex to be specified.
    def _column_cluster(x: Union[pd.Series, pd.DataFrame]) -> List:
        """Return the row indices of the cluster, for each row."""
        return [x.index.array] * len(x)

    step1_columns = step_columns(1)

    for identifier, column in zip(identifiers, step1_columns):
        dropped = df.dropna(subset=[identifier])
        grouped = dropped.groupby(identifier, sort=False)
        df[column] = grouped[identifier].transform(_column_cluster)

    # Step 2: Handle NAs to cluster papers with non-conflicting identifiers.

    # The types of 'x' and the output are a bit too complex to be specified.
    def _remove_index(x, index: int):
        """Remove the given index from the cluster."""
        if isinstance(x, ExtensionArray):
            # Use to_numpy() to pass typing checks (pandas-stubs==1.2.0.1).
            return np.delete(x, np.where(x.to_numpy() == index))
        else:
            return x

    def _handle_nans(x: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """Remove the row index from the cluster, if the row contains NAs."""
        if x.hasnans:
            return x.apply(_remove_index, index=x.name)
        else:
            return x

    step2_columns = step_columns(2)

    df[step2_columns] = df[step1_columns].apply(_handle_nans, axis=1)

    # Step 3: Cluster papers per identifier, row-wise.

    def _row_cluster(x: pd.Series) -> str:
        """Compute the smallest cluster per row, as a hashable representation."""
        intersect = functools.partial(np.intersect1d, assume_unique=True)
        cluster = functools.reduce(intersect, x.dropna())
        return np.array_str(cluster)

    df["cluster"] = df[step2_columns].apply(_row_cluster, axis=1)

    # Step 6: Generate a UUID for each cluster.

    def _generate_uuid(_) -> str:
        """Generate a UUID."""
        return str(uuid4())

    df["cluster_uuid"] = df.groupby("cluster")["cluster"].transform(_generate_uuid)

    # Step 7: Return the generated UUID per cluster of identifiers.

    return df[[*identifiers, "cluster_uuid"]]

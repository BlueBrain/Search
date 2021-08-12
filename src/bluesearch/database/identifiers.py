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
    # Step 0: Create a copy with only the columns with the identifiers.

    df = metadata[identifiers].copy()

    # Step 1: Cluster papers per identifier (i.e. column-wise).

    # The types of the output elements are a bit too complex to be specified.
    def _retrieve_indices(x: Union[pd.Series, pd.DataFrame]) -> List:
        """Return the row indices of the cluster, for each row."""
        return [x.index.array] * len(x)

    step1_columns = [f"step1_{x}" for x in identifiers]

    for identifier, column in zip(identifiers, step1_columns):
        dropped = df.dropna(subset=[identifier])
        grouped = dropped.groupby(identifier, sort=False)
        transformed = grouped[identifier].transform(_retrieve_indices)
        df[column] = transformed

    # Step 2: Let cluster papers with non-conflicting identifiers (i.e. cluster-wise).

    # The types of the input and output elements are a bit too complex to be specified.
    def _remove_index(x, index: int):
        """Remove the given index from the array, if it has more than two elements."""
        if isinstance(x, ExtensionArray) and len(x) > 2:
            # Use to_numpy() to please typing checks (pandas-stubs==1.2.0.1).
            return np.delete(x, np.where(x.to_numpy() == index))
        else:
            return x

    def _handle_nans(x: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """Remove the row index from the cluster, if the row contains NAs."""
        if x.hasnans:
            return x.apply(_remove_index, index=x.name)
        else:
            return x

    step2_columns = [f"step2_{x}" for x in identifiers]
    df[step2_columns] = df[step1_columns].apply(_handle_nans, axis=1)

    # Step 3: Compute the size of each cluster.

    step3_columns = [f"step3_{x}" for x in identifiers]
    df[step3_columns] = df[step2_columns].applymap(len, na_action="ignore")

    # Step 4: Select the smallest cluster per paper.

    offset = len(df.columns) - len(step3_columns) - len(step2_columns)

    def _identify_cluster(x: pd.Series) -> int:
        """Identify the column position of the cluster."""
        return x.argmin() + offset

    df["cluster_column"] = df[step3_columns].apply(_identify_cluster, axis=1)

    # Step 5: Create a hashable representation of each cluster.

    def _retrieve_cluster(x: pd.Series) -> str:
        """Retrieve the cluster as a string (hashable)."""
        cluster = x[x["cluster_column"]]
        return np.array_str(cluster)

    df["cluster"] = df.apply(_retrieve_cluster, axis=1)

    # Step 6: Generate a UUID for each cluster.

    def _generate_uuid(_) -> str:
        """Generate a unique identifier."""
        return str(uuid4())

    df["cluster_uuid"] = df.groupby("cluster")["cluster"].transform(_generate_uuid)

    # Step 7: Return the generated UUID per identifiers cluster.

    return df[[*identifiers, "cluster_uuid"]]

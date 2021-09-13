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

import hashlib
from typing import List, Tuple

import pandas as pd


def generate_uid(identifiers: Tuple) -> str:
    """Generate a deterministic UID for the given identifiers.

    Parameters
    ----------
    identifiers
        Values of the identifiers.

    Returns
    -------
    str
        A deterministic UID.
    """
    data = str(identifiers).encode()
    hashed = hashlib.md5(data).hexdigest()  # nosec
    return hashed


def generate_uids(metadata: pd.DataFrame, identifiers: List[str]) -> pd.DataFrame:
    """Generate UIDs for papers with multiple sources, identifiers.

    Papers with the same values for the given identifiers get the same UID.

    Missing values should have the value 'None', which is considered a value by itself.
    Then, identifiers (a, None) and identifiers (a, b) have two different UIDs.
    Papers with all given identifiers unspecified have 'None' as UID.

    The generation of the UID is deterministic.

    Parameters
    ----------
    metadata
        Paper metadata including the given identifiers.
    identifiers
        Columns of the identifiers.

    Returns
    -------
    pandas.DataFrame
        Paper metadata with a new column with the generated UIDs.
    """
    # Ignore papers without values for the given identifiers.

    df = metadata.dropna(how="all", subset=identifiers)

    # Generate a UID per paper group.

    def _uid(x: pd.Series) -> str:
        values = tuple(x.to_list())
        uid = generate_uid(values)
        return uid

    df["uid"] = df[identifiers].apply(_uid, axis=1)

    return df

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

from __future__ import annotations

import hashlib


def generate_uid(identifiers: tuple[str | None, ...]) -> str | None:
    """Generate a deterministic UID for the given paper identifiers.

    Papers with the same values for the given identifiers get the same UID.

    Missing values should have the value 'None', which is considered a value by itself.
    Then, identifiers (a, None) and identifiers (a, b) have two different UIDs.

    Papers with all given identifiers unspecified have 'None' as UID.

    Parameters
    ----------
    identifiers
        Values of the identifiers.

    Returns
    -------
    str or None
        A deterministic UID. The value is 'None' if all given identifiers are 'None'.
    """
    if all(x is None for x in identifiers):
        return None
    else:
        data = str(identifiers).encode()
        hashed = hashlib.md5(data).hexdigest()  # nosec
        return hashed

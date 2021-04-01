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

"""
This script is a patch to order deterministically model-best/vocab/strings.json.

See https://github.com/explosion/spaCy/pull/7603 for details.
"""

from pathlib import Path

import srsly
import typer
from spacy.strings import StringStore


def sort(path: Path):
    """Sort the strings from the vocabulary of a spaCy model.

    For the original code of StringStore.to_disk(), see https://github.com/explosion/spaCy/blob/53a3b967ac704ff0a67a7102ede6d916e2a4545a/spacy/strings.pyx#L219-L227.
    """
    st = StringStore().from_disk(path)
    strings = sorted(st)
    srsly.write_json(path, strings)


if __name__ == "__main__":
    typer.run(sort)

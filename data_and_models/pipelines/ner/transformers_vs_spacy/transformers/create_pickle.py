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

import argparse
import pathlib

from datasets import load_dataset
import pandas as pd
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("input", default="dataset.txt")
parser.add_argument("output", default="dataset.pkl")
args = parser.parse_args()
input_path = pathlib.Path(args.input)

sequences: List[List[str]]= []
targets: List[List[str]] = []

with input_path.open("r", encoding="utf-8") as f:
    sequence: List[str] = []
    target: List[str] = []

    all_lines = list(f.readlines())

    # Make sure the last line is a new line
    if all_lines[-1] != "\n":
        all_lines.append("\n")

    for line in all_lines:
        if line == "\n":
            sequences.append(sequence[:])
            targets.append(target[:])

            sequence.clear()
            target.clear()
            continue
         
        try:
            token, entity_type = line.split(" ")        
            entity_type = entity_type.strip("\n")
        except:
            print(f"Something went wrong: {line}")

        sequence.append(token)
        target.append(entity_type)

df = pd.DataFrame({"token": sequences, "entity_type": targets})
df.to_pickle(args.output)

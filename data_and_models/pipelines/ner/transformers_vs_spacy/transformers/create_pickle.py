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

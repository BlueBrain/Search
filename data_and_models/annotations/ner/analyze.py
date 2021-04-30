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

from pathlib import Path
from typing import Dict, Union

import pandas as pd
import typer

"""
Analyze annotations in order to clean them up for training NER models.

Valid texts are texts which are accepted texts and which contains spans (i.e. entities).

The code could either be used as a script or as a function.

As a script, it lets analyze an individual annotation file with:
```
Usage: analyze.py [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH    [required]

Options:
  --verbose     [default: False]
  --help        Show this message and exit.
```

As a function, it lets analyze several annotation files while retrieving the results:
```
from pathlib import Path

import pandas as pd
from analyze import report, METRICS

filenames = [
    "annotations5_EmmanuelleLogette_2020-06-30_raw2_Disease.jsonl",
    "annotations6_EmmanuelleLogette_2020-07-07_raw4_TaxonChebi.jsonl",
    "annotations9_EmmanuelleLogette_2020-07-08_raw6_CelltypeProtein.jsonl",
    "annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl",
    "annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl",    
    "annotations14_EmmanuelleLogette_2020-09-02_raw8_CellCompartmentDrugOrgan.jsonl",
    "annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.jsonl",
]

indexes = []
rows = []
for x in filenames:
    path = Path(x)
    indexes.append(path.stem.split("_", maxsplit=1)[0])
    rows.append(report(path, False))

table = pd.DataFrame(rows, index=indexes, columns=METRICS)
```
"""

METRICS = [
    "total texts",
    "ignored texts w/o spans",
    "ignored texts w/ spans",
    "accepted texts w/o spans",
    "valid texts",
    "% of valid texts",
    "duplicated '_input_hash'",
    "valid texts with multiple labels",
    "labels needing normalization",
]

INFO = "### INFO ###"
WARNING = "### WARNING ###"
CRITICAL = "### CRITICAL ###"
HELP = "Data could be displayed with '--verbose'."

DISPLAY = ["text", "spans"]


def report(input_path: Path, verbose: bool) -> Dict[str, Union[int, float]]:
    print(f">>> Analyze {input_path}")

    results = []

    def pprint(idx: int) -> None:
        print(f"\n{METRICS[idx]}: {results[idx]}")

    def info(idx: int) -> None:
        if results[idx] > 0:
            print(f"{INFO}\nThis might NOT be expected.\n{HELP}")

    def warn(idx: int) -> None:
        if results[idx] > 0:
            print(f"{WARNING}\nThis needs to be DISCUSSED.\n{HELP}")

    def critical(idx: int) -> None:
        if results[idx] > 0:
            print(f"{CRITICAL}\nThis needs to be INVESTIGATED.\n{HELP}")

    def debug(df: pd.DataFrame) -> None:
        if verbose:
            print(df[DISPLAY])

    def debug_labels(df: pd.DataFrame) -> None:
        counted = df.groupby("spans", as_index=False).count()
        print(counted.to_string(header=False, index=False))

    def normalize(series: pd.Series) -> pd.Series:
        # Change to adapt to new discovered normalization to apply.
        return series.str.upper()

    df0 = pd.read_json(input_path, lines=True)
    results.append(len(df0))
    pprint(0)

    # Sometimes, spans is missing.
    df0.spans = df0.spans.map(
        lambda x: [y["label"] for y in x] if isinstance(x, list) and x else None
    )

    df1 = df0[(df0.answer == "ignore") & df0.spans.isna()]
    results.append(len(df1))
    pprint(1)
    info(1)
    debug(df1)

    df2 = df0[(df0.answer == "ignore") & df0.spans.notna()]
    results.append(len(df2))
    pprint(2)
    warn(2)
    debug(df2)

    df3 = df0[(df0.answer == "accept") & df0.spans.isna()]
    results.append(len(df3))
    pprint(3)
    warn(3)
    debug(df3)

    df4 = df0[(df0.answer == "accept") & df0.spans.notna()]
    results.append(len(df4))
    results.append(float(f"{(results[4] / results[0])*100:.2f}"))
    pprint(4)
    pprint(5)
    debug(df4)

    results.append(len(df0) - df0._input_hash.nunique())
    pprint(6)
    if results[6] > 0:
        print(f"{CRITICAL}\nThis needs to be INVESTIGATED.\n{HELP}")
        print("NB: Following counts might NOT be accurate because of the duplicate(s).")
    if verbose:
        print(df0[df0.duplicated("_input_hash", keep=False)].T)

    exploded = df4[["_input_hash", "spans"]].explode("spans")
    grouped = exploded.drop_duplicates().groupby("_input_hash").count()

    df7 = grouped[grouped.spans != 1]
    results.append(len(df7))
    pprint(7)
    info(7)
    if verbose:
        df7_df4 = df7.merge(df4, on="_input_hash", suffixes=["_count", ""])
        print(df7_df4[[*DISPLAY, "spans_count"]])

    spans_normalized = normalize(exploded.spans)

    results.append(exploded.spans.nunique() - spans_normalized.nunique())
    pprint(8)
    critical(8)
    if verbose:
        print("\nlabels for all texts after normalization:")
        debug_labels(exploded)

    print("\nlabels for all texts before normalization:")
    exploded.spans = spans_normalized
    debug_labels(exploded)

    print("")

    return results


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    verbose: bool = typer.Option(False, "--verbose"),
):
    report(input_path, verbose)


if __name__ == "__main__":
    typer.run(main)

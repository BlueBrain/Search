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

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import pandas as pd

"""
This code lets analyze annotations in order to clean them for training NER models.

When there are duplicated '_input_hash', the following counts might NOT be accurate:
  - accepted texts w/ multiple labels,
  - labels needing normalization.
Indeed, the computation of these counts relies on '_input_hash'.

This code could be used either as a script or as a function.

As a script, it lets analyze an individual annotation file with a detailed report.

As a function, it lets analyze several annotation files with a summary table:
```
from pathlib import Path

import pandas as pd
from analyze import report, METRICS

filenames = [
    "annotations5_EmmanuelleLogette_2020-06-30_raw2_Disease.jsonl",
    "annotations6_EmmanuelleLogette_2020-07-07_raw4_TaxonChebi.jsonl",
    "annotations9_EmmanuelleLogette_2020-07-08_raw6_CelltypeProtein.jsonl",
    "annotations14_EmmanuelleLogette_2020-09-02_raw8_CellCompartmentDrugOrgan.jsonl",
    "annotations15_EmmanuelleLogette_2020-09-22_raw9_Pathway.jsonl",
    "annotations10_EmmanuelleLogette_2020-08-28_raw1_raw5_10EntityTypes.jsonl",
    "annotations12_EmmanuelleLogette_2020-08-28_raw7_10EntityTypes.jsonl",
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
    "accepted texts w/ spans",
    "accepted texts w/ spans (%)",
    "duplicated '_input_hash'",
    "accepted texts w/ multiple labels",
    "labels needing normalization",
]


def report(input_path: Path, verbose: bool) -> List[Union[int, float]]:
    print(f"Analyzing {input_path}...")

    results = []

    def msg(idx: int, text: str) -> None:
        if results[idx] > 0:
            print(text)

    def info(idx: int) -> None:
        msg(idx, f"### INFO ### This might not be expected.")

    def warn(idx: int) -> None:
        msg(idx, f"### WARNING ### There might be an issue.")

    def critical(idx: int) -> None:
        msg(idx, f"### CRITICAL ### There is an issue to be investigated.")

    def presult(idx: int) -> None:
        print(f"\n{METRICS[idx]}: {results[idx]}")

    def pdataframe(df: pd.DataFrame, columns: List[str] = ["text", "spans"]) -> None:
        if verbose:
            limit = 10
            if len(df) > limit:
                print(f"(only the first 10 rows are displayed)")
            print(df[columns].head(limit))

    def plabels(df: pd.DataFrame) -> None:
        counted = df.groupby("spans", as_index=False).count()
        print(counted.to_string(header=False, index=False))

    def normalize(series: pd.Series) -> pd.Series:
        # Change to adapt to new discovered normalization to apply.
        return series.str.upper()

    # Total texts.
    df0 = pd.read_json(input_path, lines=True)
    results.append(len(df0))
    presult(0)

    # Sometimes, 'spans' is missing.
    df0.spans = df0.spans.map(
        lambda x: [y["label"] for y in x] if isinstance(x, list) and x else None
    )

    # Ignored texts w/o spans.
    df1 = df0[(df0.answer == "ignore") & df0.spans.isna()]
    results.append(len(df1))
    presult(1)
    info(1)
    pdataframe(df1)

    # Ignored texts w/ spans.
    df2 = df0[(df0.answer == "ignore") & df0.spans.notna()]
    results.append(len(df2))
    presult(2)
    warn(2)
    pdataframe(df2)

    # Accepted texts w/o spans.
    df3 = df0[(df0.answer == "accept") & df0.spans.isna()]
    results.append(len(df3))
    presult(3)
    info(3)
    pdataframe(df3)

    # Accepted texts w/ spans.
    df4 = df0[(df0.answer == "accept") & df0.spans.notna()]
    results.append(len(df4))
    presult(4)
    pdataframe(df4)

    # Accepted texts w/ spans (%).
    results.append(float(f"{(results[4] / results[0])*100:.2f}"))
    presult(5)

    # Duplicated '_input_hash'.
    df6 = df0[df0.duplicated("_input_hash", keep=False)].sort_values("_input_hash")
    results.append(df6._input_hash.nunique())
    presult(6)
    critical(6)
    pdataframe(df6, df6.columns)

    exploded = df4[["_input_hash", "spans"]].explode("spans")
    grouped = exploded.drop_duplicates().groupby("_input_hash").count()

    # Accepted texts w/ multiple labels.
    df7 = grouped[grouped.spans != 1]
    results.append(len(df7))
    presult(7)
    info(7)
    df7_df4 = df7.merge(df4, on="_input_hash", suffixes=["_count", ""])
    pdataframe(df7_df4, ["text", "spans", "spans_count"])

    spans_normalized = normalize(exploded.spans)

    # Labels needing normalization.
    results.append(exploded.spans.nunique() - spans_normalized.nunique())
    presult(8)
    critical(8)
    if verbose:
        print("\nlabels before normalization:")
        plabels(exploded)

    print("\nlabels after normalization:")
    exploded.spans = spans_normalized
    plabels(exploded)

    print(f"\n...analyzed {input_path}")

    return results


parser = ArgumentParser()
parser.add_argument(
    "input_path",
    type=Path,
    help="path of the annotations exported with Prodigy (.jsonl file)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="display the concerned data for each summary metrics",
)
args = parser.parse_args()


def main():
    report(args.input_path, args.verbose)


if __name__ == "__main__":
    main()

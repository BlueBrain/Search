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
import json
from functools import reduce
from pprint import pprint

import pandas as pd
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from seqeval.metrics import performance_measure
from seqeval.scheme import IOB2

from bluesearch.mining.eval import ner_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("input_df_pkl")
    args = parser.parse_args()

    test_df = pd.read_pickle(args.input_df_pkl)
    y_pred = []
    with open(args.predictions) as fp:
        for line in fp:
            y_pred += line.strip().split()

    y_true = reduce(lambda acc, l: acc + l, test_df.entity_type, [])
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    print(len(y_true))
    print(len(y_pred))

    print("Token level")
    eval_d = ner_report(y_true, y_pred, mode="token", return_dict=True)
    eval_d = dict(eval_d["PATHWAY"])
    with open("pathway_metrics_token.json", "w") as fp:
        json.dump(eval_d, fp)
        fp.write("\n")
    pprint(eval_d)

    print("Entity level")
    y_pred_corr = pd.Series(correct_iob(y_pred))
    eval_d = ner_report(y_true, y_pred_corr, mode="entity", return_dict=True)
    eval_d = dict(eval_d["PATHWAY"])
    with open("pathway_metrics_entity.json", "w") as fp:
        json.dump(eval_d, fp)
        fp.write("\n")
    pprint(eval_d)

    print("Seqeval")
    y_true = list(test_df.entity_type)
    y_pred = []
    with open(args.predictions) as fp:
        for line in fp:
            y_pred.append(line.strip().split())

    from collections import Counter
    c = Counter()
    for x in y_true:
        c.update(x)
    print(c)
    total = 0
    for s1, s2 in zip(y_true, y_pred):
        total += sum(t1 == t2 for t1, t2 in zip(s1, s2))
    acc = total / sum(len(s) for s in y_true)
    print("acc:", acc)
    print("acc_score:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, scheme=IOB2, mode="strict"))
    print(performance_measure(y_true, y_pred))


def correct_iob(y_pred):
    pred = list(y_pred)
    for i in range(len(pred)):
        if pred[i].startswith("I-") and (i == 0 or pred[i-1] == "O"):
            pred[i] = "B-" + pred[i][2:]
    return pred

if __name__ == "__main__":
    main()

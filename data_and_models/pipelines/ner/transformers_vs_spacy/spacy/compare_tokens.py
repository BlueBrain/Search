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

import pandas as pd
import spacy


nlp = spacy.load("en_core_web_lg", disable=["vocab", "ner"])


def unroll_rows(df):
    return pd.concat([pd.DataFrame(row.to_dict()) for i, row in df.iterrows()])


def poor_venn(set1, set2):
    print(f"[ {len(set1 - set2)} | {len(set1 & set2)} | {len(set2 - set1)} ]")


def lemma(word):
    return next(iter(nlp(word.lower()))).lemma_


df_train = pd.read_pickle("../transformers/train_data.pkl")
df_test = pd.read_pickle("df_test_pred.pkl")

df_train_flat = unroll_rows(df_train)

train_entities = set(df_train_flat.token[df_train_flat.entity_type != "O"])
test_entities = set(df_test.text[df_test["class"].isin(["B-PATHWAY", "I-PATHWAY"])])
pred_entities = set(df_test.text[df_test.class_pred.isin(["B-PATHWAY", "I-PATHWAY"])])

train_entities = set(map(lemma, train_entities))
test_entities = set(map(lemma, test_entities))
pred_entities = set(map(lemma, pred_entities))

print("{train, test, pred} = Unique token lemmata in the corresponding sets with an entity type that is not 'O'")
print()

print("train - test")
print(sorted(train_entities - test_entities))
print()

print("train - pred")
print(sorted(train_entities - pred_entities))
print()

print("test - train")
print(sorted(test_entities - train_entities))
print()

print("pred - train")
print(sorted(pred_entities - train_entities))
print()

print("len(train) =", len(train_entities))
print("len(test) =", len(test_entities))
print("len(pred) =", len(pred_entities))
print()

print("VENN: train vs. test")
poor_venn(train_entities, test_entities)
print("VENN: train vs. pred")
poor_venn(train_entities, pred_entities)
print("VENN: test vs. pred")
poor_venn(test_entities, pred_entities)
print()

print("How many of the unseen tokens were predicted?")
seen = test_entities & train_entities
unseen = test_entities - train_entities
print(f"Out of {len(unseen)} unseen tokens {len(unseen & pred_entities)} were predicted")
print(f"Out of {len(seen)} seen tokens {len(seen & pred_entities)} were predicted")
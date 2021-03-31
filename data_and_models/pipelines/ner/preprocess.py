# Adapted from https://github.com/explosion/projects/blob/v3/tutorials/ner_drugs/scripts/preprocess.py.

# MIT License
#
# Copyright (c) 2020 ExplosionAI GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
This script splits and converts annotations from Prodigy <= v1.10.x.

First, this script splits the annotations into train and dev corpora.
Second, this script converts .jsonl files from Prodigy to .spacy files.

The output is 2 files. Each are named as the input file but with the extension changed.
The train corpus has the extension ".train.spacy". The dev corpus, ".dev.spacy".
"""

from pathlib import Path

import spacy
import srsly
import typer
import yaml
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces


def main(input_path: Path = typer.Argument(..., exists=True, dir_okay=False)):
    print("Read params.yaml...")
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    dev_size = params["train"]["corpora"]["dev_size"]
    shuffle_seed = params["train"]["corpora"]["shuffle_seed"]
    print(f"...read dev_size={dev_size}, shuffle_seed={shuffle_seed}")

    print("Read annotations...")
    corpus = list(srsly.read_jsonl(input_path))
    print(f"...read {len(corpus)} texts")

    print("Convert into documents...")
    docs = []
    nlp = spacy.blank("en")
    for eg in corpus:
        if eg["answer"] != "accept":
            continue
        tokens = [token["text"] for token in eg["tokens"]]
        words, spaces = get_words_and_spaces(tokens, eg["text"])
        doc = Doc(nlp.vocab, words=words, spaces=spaces)
        doc.ents = [
            doc.char_span(s["start"], s["end"], label=s["label"])
            for s in eg.get("spans", [])
        ]
        docs.append(doc)
    print(f"...converted {len(docs)} documents")

    print("Split into train and dev...")
    train, dev = train_test_split(
        docs, test_size=dev_size, random_state=shuffle_seed, shuffle=True
    )
    print(f"...split into {len(train)} train and {len(dev)} dev documents")

    print("Write serialized documents...")
    for split, data in [("train", train), ("dev", dev)]:
        output_path = input_path.with_suffix(f".{split}.spacy")
        doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"], docs=data)
        doc_bin.to_disk(output_path)
        print(f"...wrote {output_path}")


if __name__ == "__main__":
    typer.run(main)

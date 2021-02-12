"""Classes and functions for relation extraction."""

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

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd


class REModel(ABC):
    """Abstract interface for relationship extraction models.

    Inspired by SciBERT.
    """

    @property
    @abstractmethod
    def classes(self):
        """Names of supported relation classes.

        Returns
        -------
        list of str
            Names of supported relation classes.
        """

    @abstractmethod
    def predict_probs(self, annotated_sentence):
        """Relation probabilities between subject and object.

        Predict per-class probabilities for the relation between subject and
        object in an annotated sentence.

        Parameters
        ----------
        annotated_sentence : str
            Sentence with exactly 2 entities being annotated accordingly.
            For example "<< Cytarabine >> inhibits [[ DNA polymerase ]]."

        Returns
        -------
        relation_probs : pd.Series
            Per-class probability vector. The index contains the class names,
            the values are the probabilities.
        """

    def predict(self, annotated_sentence, return_prob=False):
        """Predict most likely relation between subject and object.

        Parameters
        ----------
        annotated_sentence : str
            Sentence with exactly 2 entities being annotated accordingly.
            For example "<< Cytarabine >> inhibits [[ DNA polymerase ]]."
        return_prob : bool, optional
            If True also returns the confidence of the predicted relation.

        Returns
        -------
        relation : str
            Relation type.
        prob : float, optional
            Confidence of the predicted relation.
        """
        probs = self.predict_probs(annotated_sentence)
        relation = probs.idxmax()
        if return_prob:
            prob = probs.max()
            return relation, prob
        else:
            return relation

    @property
    @abstractmethod
    def symbols(self):
        """Generate dictionary mapping the two entity types to their annotation symbols.

        General structure: {'ENTITY_TYPE': ('SYMBOL_LEFT', 'SYMBOL_RIGHT')}
        Specific example: {'GGP': ('[[ ', ' ]]'), 'CHEBI': ('<< ', ' >>')}

        Make sure that left and right symbols are not identical.
        """


def annotate(doc, sent, ent_1, ent_2, etype_symbols):
    """Annotate sentence given two entities.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The entire document (input text). Note that spacy uses it for
        absolute referencing.
    sent : spacy.tokens.Span
        One sentence from the `doc` where we look for relations.
    ent_1 : spacy.tokens.Span
        The first entity in the sentence. One can get its type by using the
        `label_` attribute.
    ent_2 : spacy.tokens.Span
        The second entity in the sentence. One can get its type by using the
        `label_` attribute.
    etype_symbols : dict or defaultdict
        Keys represent different entity types ("GGP", "CHEBI") and the values
        are tuples of size 2. Each of these tuples represents the starting
        and ending symbol to wrap the recognized entity with. Each ``REModel``
        has the `symbols` property that encodes how its inputs should be annotated.

    Returns
    -------
    result : str
        String representing an annotated sentence created out of the original one.

    Notes
    -----
    The implementation is non-trivial because an entity can span multiple words.
    """
    # checks
    if ent_1 == ent_2:
        raise ValueError("One needs to provide two separate entities.")

    if not (
        sent.start <= ent_1.start <= ent_1.end <= sent.end
        and sent.start <= ent_2.start <= ent_2.end <= sent.end
    ):
        raise ValueError("The provided entities are outside of the given sentence.")

    etype_1 = ent_1.label_
    etype_2 = ent_2.label_

    if not isinstance(etype_symbols, defaultdict) and not (
        etype_1 in etype_symbols and etype_2 in etype_symbols
    ):
        raise ValueError(
            "Please specify the special symbols for both of the entity types."
        )

    tokens = []
    i = sent.start
    while i < sent.end:
        new_tkn = " "  # hack to keep the punctuation nice

        if ent_1.start == i:
            start, end = ent_1.start, ent_1.end
            new_tkn += (
                etype_symbols[etype_1][0]
                + doc[start:end].text
                + etype_symbols[etype_1][1]
            )

        elif ent_2.start == i:
            start, end = ent_2.start, ent_2.end
            new_tkn += (
                etype_symbols[etype_2][0]
                + doc[start:end].text
                + etype_symbols[etype_2][1]
            )

        else:
            start, end = i, i + 1
            new_tkn = doc[i].text if doc[i].is_punct else new_tkn + doc[i].text

        tokens.append(new_tkn)
        i += end - start

    return "".join(tokens).strip()


class ChemProt(REModel):
    """Pretrained model extracting 13 relations between chemicals and proteins.

    This model supports the following entity types:
        - "GGP"
        - "CHEBI"

    Attributes
    ----------
    model_ : allennlp.predictors.text_classifier.TextClassifierPredictor
        The actual model in the backend.

    Notes
    -----
    This model depends on a package named `scibert` which is not specified in
    the `setup.py` since it introduces dependency conflicts. One can
    install it manually with the following command.

    .. code-block:: bash

        pip install git+https://github.com/allenai/scibert

    Note that `import scibert` has a side effect of registering the
    "text_classifier" model with `allennlp`. This is done via applying a
    decorator to a class. For more details see

    https://github.com/allenai/scibert/blob/06793f77d7278898159ed50da30d173cdc8fdea9/scibert/models/text_classifier.py#L14
    """

    def __init__(self, model_path):
        # Note: SciBERT is imported but unused. This is because the import has
        # a side-effect of registering the SciBERT model, which we use later on.
        import scibert  # NOQA
        from allennlp.predictors import Predictor

        self.model_ = Predictor.from_path(model_path, predictor_name="text_classifier")

    @property
    def classes(self):
        """Names of supported relation classes."""
        return [
            "INHIBITOR",
            "SUBSTRATE",
            "INDIRECT-DOWNREGULATOR",
            "INDIRECT-UPREGULATOR",
            "ACTIVATOR",
            "ANTAGONIST",
            "PRODUCT-OF",
            "AGONIST",
            "DOWNREGULATOR",
            "UPREGULATOR",
            "AGONIST-ACTIVATOR",
            "SUBSTRATE_PRODUCT-OF",
            "AGONIST-INHIBITOR",
        ]

    @property
    def symbols(self):
        """Symbols for annotation."""
        return {"GGP": ("[[ ", " ]]"), "CHEBI": ("<< ", " >>")}

    def predict_probs(self, annotated_sentence):
        """Predict probabilities for the relation."""
        return pd.Series(
            self.model_.predict(sentence=annotated_sentence)["class_probs"],
            index=self.classes,
        )


class StartWithTheSameLetter(REModel):
    """Check whether two entities start with the same letter (case insensitive).

    This relation is symmetric and works on any entity type.
    """

    @property
    def classes(self):
        """Names of supported relation classes."""
        return ["START_WITH_SAME_LETTER", "START_WITH_DIFFERENT_LETTER"]

    def predict_probs(self, annotated_sentence):
        """Predict probabilities for the relation."""
        left_symbol, _ = self.symbols["anything"]
        s_len = len(left_symbol)

        ent_1_first_letter_ix = annotated_sentence.find(left_symbol) + s_len
        ent_2_first_letter_ix = (
            annotated_sentence.find(left_symbol, ent_1_first_letter_ix + 1) + s_len
        )

        ent_1_first_letter = annotated_sentence[ent_1_first_letter_ix]
        ent_2_first_letter = annotated_sentence[ent_2_first_letter_ix]

        if ent_1_first_letter.lower() == ent_2_first_letter.lower():
            return pd.Series([1, 0], index=self.classes)
        else:
            return pd.Series([0, 1], index=self.classes)

    @property
    def symbols(self):
        """Symbols for annotation."""
        return defaultdict(lambda: ("[[ ", " ]]"))

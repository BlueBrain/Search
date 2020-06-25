"""Relation Extraction."""
from abc import ABC, abstractmethod
from collections import defaultdict

from allennlp.predictors import Predictor
import pandas as pd
from scibert.models.text_classifier import TextClassifier  # noqa


class REModel(ABC):
    """Abstract interface for relationship extraction models.

    Inspired by SciBERT.
    """

    @property
    @abstractmethod
    def classes(self):
        """list[str]: Names of supported relation classes."""

    @abstractmethod
    def predict_probs(self, annotated_sentence):
        """Predict per-class probabilities for the relation between subject and object in an annotated sentence.

        Parameters
        ----------
        annotated_sentence : str
            Sentence with exactly 2 entities being annotated accordingly.
            For example "<< Cytarabine >> inhibits [[ DNA polymerase ]]."

        Returns
        -------
        relation_probs : pd.Series
            Per-class probability vector. The index contains the class names, the values are the probabilities.
        """

    def predict(self, annotated_sentence, return_prob=False):
        """Predict most likely relation between subject and object in an annotated sentence.

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
        Specific example: {'GGP': ('[[ ', ' ]]'),
                           'CHEBI': ('<< ', ' >>')}

        Make sure that left and right symbols are not identical.
        """


def annotate(doc, sent, ent_1, ent_2, etype_symbols):
    """Annotate sentence given two entities.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The entire document (input text). Note that spacy uses it for absolute referencing.
    sent : spacy.tokens.Span
        One sentence from the `doc` where we look for relations.
    ent_1 : pd.Series
        A single row of the dataframe of extracted entities, holding info on first entity in the sentence.
    ent_2 : pd.Series
        A single row of the dataframe of extracted entities, holding info on second entity in the sentence.

    etype_symbols: dict or defaultdict
        Keys represent different entity types ("GGP", "CHEBI") and the values are tuples of size 2.
        Each of these tuples represents the starting and ending symbol to wrap the recognized entity with.
        Each ``REModel`` has the `symbols` property that encodes how its inputs should be annotated.

    Returns
    -------
    result : str
        String representing an annotated sentence created out of the original one.

    Notes
    -----
    The implementation is non-trivial because an entity can span multiple words.
    """
    # checks
    if (ent_1.start_char == ent_2.start_char) and (ent_1.end_char == ent_2.end_char):
        raise ValueError('One needs to provide two separate entities.')

    if not (sent.start_char <= ent_1.start_char <= ent_1.end_char <= sent.end_char
            and
            sent.start_char <= ent_2.start_char <= ent_2.end_char <= sent.end_char):
        raise ValueError('The provided entities are outside of the given sentence.')

    etype_1 = ent_1.entity_type
    etype_2 = ent_2.entity_type

    if not isinstance(etype_symbols, defaultdict) and not (etype_1 in etype_symbols and etype_2 in etype_symbols):
        raise ValueError('Please specify the special symbols for both of the entity types.')

    # Add symbols at the proper place, for each of the entities
    text = doc.text
    e1, e2 = sorted([ent_1, ent_2], key=lambda e: e.start_char)

    result = text[sent.start_char:e1.start_char] + \
        etype_symbols[e1.entity_type][0] + \
        e1.entity + \
        etype_symbols[e1.entity_type][1] + \
        text[e1.end_char:e2.start_char] + \
        etype_symbols[e2.entity_type][0] + \
        e2.entity + \
        etype_symbols[e2.entity_type][1] + \
        text[e2.end_char:sent.end_char]

    return result


class ChemProt(REModel):
    """Pretrained model extracting 13 relations between chemicals and proteins.

    This model supports the following entity types:
        - "GGP"
        - "CHEBI"

    Attributes
    ----------
    model_ : allennlp.predictors.text_classifier.TextClassifierPredictor
        The actual model in the backend.
    """

    def __init__(self, model_path):
        self.model_ = Predictor.from_path(model_path, predictor_name='text_classifier')

    @property
    def classes(self):
        """Names of supported relation classes."""
        return [
            'INHIBITOR',
            'SUBSTRATE',
            'INDIRECT-DOWNREGULATOR',
            'INDIRECT-UPREGULATOR',
            'ACTIVATOR',
            'ANTAGONIST',
            'PRODUCT-OF',
            'AGONIST',
            'DOWNREGULATOR',
            'UPREGULATOR',
            'AGONIST-ACTIVATOR',
            'SUBSTRATE_PRODUCT-OF',
            'AGONIST-INHIBITOR']

    @property
    def symbols(self):
        """Symbols for annotation."""
        return {'GGP': ('[[ ', ' ]]'),
                'CHEBI': ('<< ', ' >>')}

    def predict_probs(self, annotated_sentence):
        """Predict probabilities for the relation."""
        return pd.Series(self.model_.predict(sentence=annotated_sentence)['class_probs'], index=self.classes)


class StartWithTheSameLetter(REModel):
    """Check whether two entities start with the same letter (case insensitive).

    This relation is symmetric and works on any entity type.
    """

    @property
    def classes(self):
        """Names of supported relation classes."""
        return ['START_WITH_SAME_LETTER', 'START_WITH_DIFFERENT_LETTER']

    def predict_probs(self, annotated_sentence):
        """Predict probabilities for the relation."""
        left_symbol, _ = self.symbols['anything']
        s_len = len(left_symbol)

        ent_1_first_letter_ix = annotated_sentence.find(left_symbol) + s_len
        ent_2_first_letter_ix = annotated_sentence.find(left_symbol, ent_1_first_letter_ix + 1) + s_len

        ent_1_first_letter = annotated_sentence[ent_1_first_letter_ix]
        ent_2_first_letter = annotated_sentence[ent_2_first_letter_ix]

        if ent_1_first_letter.lower() == ent_2_first_letter.lower():
            return pd.Series([1, 0], index=self.classes)
        else:
            return pd.Series([0, 1], index=self.classes)

    @property
    def symbols(self):
        """Symbols for annotation."""
        return defaultdict(lambda: ('[[ ', ' ]]'))

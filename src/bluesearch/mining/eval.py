"""Classes and functions for evaluating mining models predictions."""

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

import copy
import json
import string
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from spacy.tokens import Doc


def _check_consistent_iob(iob_true: pd.Series, iob_pred: pd.Series) -> None:
    """Check that iob_true and iob_pred are consistent (length and format).

    This function raises a ValueError if any of the targets uses an annotation
    format different from IOB2 (see [1] for the definition of this format).

    Parameters
    ----------
    iob_true :
         Ground truth (correct) IOB2 annotations.
    iob_pred :
        Predicted IOB2 annotations.

    Raises
    ------
    ValueError :
        If annotations have inconsistent length or annotation format.

    References
    ----------
    [1] Sang et al. 1999, "Representing Text Chunks", https://arxiv.org/abs/cs/9907006
    """
    if len(iob_true) != len(iob_pred):
        raise ValueError(
            f"Found target variables with inconsistent numbers of samples: "
            f"iob_true = {len(iob_true)}, iob_pred = {len(iob_pred)}."
        )

    for x in (iob_true, iob_pred):
        if not (
            x.str.startswith("B-") | x.str.startswith("I-") | x.str.startswith("O")
        ).all():
            errs = x[
                ~(
                    x.str.startswith("B-")
                    | x.str.startswith("I-")
                    | x.str.startswith("O")
                )
            ]
            raise ValueError(
                f"Annotations are not in IOB2 format! Each label must be one of\n"
                f"    'O', 'B-<entity_type>', 'I-<entity_type>'\n"
                f"but found the following inconsistent labels:\n"
                f"    {', '.join(repr(e) for e in errs.unique())}."
            )

        etypes = unique_etypes(x)
        x_prev = x.shift(periods=1).fillna("O", inplace=False)
        for etype in etypes:
            if (
                x.isin([f"I-{etype}"]) & ~x_prev.isin([f"B-{etype}", f"I-{etype}"])
            ).any():
                raise ValueError(
                    f"Annotations are not in IOB2 format! Label 'I-{etype}' "
                    f"should follow one of 'B-{etype}' or 'I-{etype}'."
                )


def annotations2df(annots_files, not_entity_symbol="O"):
    """Convert prodigy annotations in JSONL format into a pd.DataFrame.

    Parameters
    ----------
    annots_files : str, list of str, path or list of path
        Name of the annotation file(s) to load.
    not_entity_symbol : str
        A symbol to use for tokens that are not an entity.

    Returns
    -------
    final_table : pd.DataFrame
        Each row represents one token, the columns are 'source', 'sentence_id', 'class',
        'start_char', end_char', 'id', 'text'.
    """
    final_table_rows = []

    if isinstance(annots_files, list):
        final_tables = [annotations2df(ann, not_entity_symbol) for ann in annots_files]
        final_table = pd.concat(final_tables, ignore_index=True)
        return final_table
    elif not (isinstance(annots_files, str) or isinstance(annots_files, Path)):
        raise TypeError(
            "Argument 'annots_files' should be a string or an " "iterable of strings!"
        )

    with open(annots_files) as f:
        for row in f:
            content = json.loads(row)

            if content["answer"] != "accept":
                continue

            # annotations for the sentence: list of dict (or empty list)
            spans = content.get("spans", [])

            classes = {}
            for ent in spans:
                for ix, token_ix in enumerate(
                    range(ent["token_start"], ent["token_end"] + 1)
                ):
                    ent_label = ent["label"].upper()

                    classes[token_ix] = "{}-{}".format(
                        "B" if ix == 0 else "I", ent_label
                    )

            for token in content["tokens"]:
                final_table_rows.append(
                    {
                        "source": content["meta"]["source"],
                        "class": classes.get(token["id"], not_entity_symbol),
                        "start_char": token["start"],
                        "end_char": token["end"],
                        "id": token["id"],
                        "text": token["text"],
                    }
                )

    final_table = pd.DataFrame(final_table_rows)

    return final_table


def spacy2df(
    spacy_model,
    ground_truth_tokenization,
    not_entity_symbol="O",
    excluded_entity_type="NaE",
):
    """Turn NER of a spacy model into a pd.DataFrame.

    Parameters
    ----------
    spacy_model : spacy.language.Language
        Spacy model that will be used for NER, EntityRuler and Tagger
        (not tokenization). Note that a Tagger might be necessary for
        tagger EntityRuler.
    ground_truth_tokenization : list
        List of str (words) representing the ground truth tokenization.
        This will guarantee that the ground truth dataframe will be aligned
        with the prediction dataframe.
    not_entity_symbol : str
        A symbol to use for tokens that are not a part of any entity.
        Note that this symbol will be used for all tokens for which the
        `ent_iob_` attribute of `spacy.Token` is equal to "O".
    excluded_entity_type : str or None
        Entity type that is going to be automatically excluded. Note that
        it is different from `not_entity_symbol` since it corresponds to the
        `label_` attribute of ``spacy.Span`` objects. If None, then no
        exclusion will be taking place.

    Returns
    -------
    pd.DataFrame
        Each row represents one token, the columns are 'text' and 'class'.

    Notes
    -----
    One should run the `annotations2df` first in order to obtain the
    `ground_truth_tokenization`. If it is the case then
    `ground_truth_tokenization=prodigy_table['text'].to_list()`.
    """
    doc = Doc(spacy_model.vocab, words=ground_truth_tokenization)

    for _, pipe in spacy_model.pipeline:
        doc = pipe(doc)

    doc.ents = tuple(
        [
            e
            for e in doc.ents
            if excluded_entity_type is None or e.label_ != excluded_entity_type
        ]
    )

    all_rows = []
    for token in doc:

        if token.ent_iob_ == "O":
            all_rows.append(
                {
                    "class": not_entity_symbol,
                    "text": token.text,
                }
            )
        else:
            all_rows.append(
                {
                    "class": "{}-{}".format(token.ent_iob_, token.ent_type_),
                    "text": token.text,
                }
            )

    return pd.DataFrame(all_rows)


def remove_punctuation(df):
    """Remove punctuation from a dataframe with tokens and entity annotations.

    Important: this function should be called only after all the annotations
    have been loaded by calling `annotations2df()` and `spacy2df()`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tokens and annotations, can be generated calling
        `annotations2df()` and `spacy2df()`. Should include a column
        "text" containing one token per row, and one or more columns of
        annotations in IOB2 format named as "class_XXX".

    Returns
    -------
    df_cleaned : pd.DataFrame
        DataFrame with removed punctuation.
    """
    is_punctuation = df["text"].isin(list(string.punctuation))

    annotations_cols = [col for col in df.columns if col.startswith("class_")]
    for col in annotations_cols:
        for idx in df.index[is_punctuation & df[col].str.startswith("B-")]:
            i = idx
            while i < len(df) - 1 and df.iloc[i]["text"] in list(string.punctuation):
                i += 1
            df.iloc[i, df.columns.get_loc(col)] = (
                "B" + df.iloc[i][col][1:] if df.iloc[i][col] != "O" else "O"
            )

    df_cleaned = df[~is_punctuation].reset_index(drop=True)
    return df_cleaned


def unique_etypes(iob, return_counts=False, mode="entity"):
    """Return the sorted unique entity types for annotations in IOB2 format.

    Parameters
    ----------
    iob : pd.Series[str]
        Annotations in the IOB2 format. Elements of the pd.Series should
        be either 'O', 'B-ENTITY_TYPE', or 'I-ENTITY_TYPE', where
        'ENTITY_TYPE' is the name of some entity type.
    return_counts : bool, optional
        If True, also return the number of times each unique entity
        type appears in the input.
    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an
        'entity' can span several tokens.

    Returns
    -------
    unique : list[str]
        The sorted unique entity types.
    unique_counts : list[int], optional
        The number of times each of the unique entity types comes up in
        the input. Only provided if `return_counts` is True.
    """
    unique = sorted(
        {
            etype.replace("B-", "").replace("I-", "")
            for etype in iob.unique()
            if etype != "O"
        }
    )
    if not return_counts:
        return unique
    else:
        if mode == "entity":
            unique_counts = [(iob == f"B-{etype}").sum().item() for etype in unique]
        elif mode == "token":
            unique_counts = [
                (iob.isin([f"B-{etype}", f"I-{etype}"])).sum().item()
                for etype in unique
            ]
        else:
            raise ValueError(f"Mode '{mode}' is not available.")
        return unique, unique_counts


def iob2idx(iob, etype):
    """Retrieve start and end indices of entities from annotations in IOB2 format.

    Parameters
    ----------
    iob : pd.Series[str]
        Annotations in the IOB2 format. Elements of the pd.Series should be
        either 'O', 'B-ENTITY_TYPE', or 'I-ENTITY_TYPE', where 'ENTITY_TYPE'
        is the name of some entity type.
    etype : str
        Name of the entity type of interest.

    Returns
    -------
    idxs : pd.DataFrame[int, int]
        Dataframe with 2 columns, 'start' and 'end', representing start
        and end position of the entities of the specified entity type.
    """
    b_symbol = f"B-{etype}"
    i_symbol = f"I-{etype}"

    iob_next = iob.shift(periods=-1)

    data_dict = {
        "start": iob.index[iob == b_symbol],
        "end": iob.index[iob.isin([b_symbol, i_symbol]) & (iob_next != i_symbol)],
    }

    idxs = pd.DataFrame(data=data_dict)
    return idxs


def idx2text(tokens, idxs):
    """Retrieve entities text from a list of tokens and start and end indices.

    Parameters
    ----------
    tokens : pd.Series[str]
        Tokens obtained from tokenization of a text.
    idxs : pd.Series[int, int]
        Dataframe with 2 columns, 'start' and 'end', representing start
        and end position of the entities of the specified entity type.

    Returns
    -------
    texts : pd.Series[str]
        Texts of each entity identified by the indices provided in input.
    """
    return pd.Series(
        [" ".join(tokens[s : e + 1]) for s, e in zip(idxs["start"], idxs["end"])],
        index=idxs.index,
        dtype="str",
    )


def ner_report(
    iob_true: pd.Series,
    iob_pred: pd.Series,
    mode: str = "entity",
    etypes_map: Optional[dict] = None,
    return_dict: bool = False,
) -> Union[str, OrderedDict]:
    """Build a summary report showing the main ner evaluation metrics.

    Evaluation is performed according to the definitions of "errors" from [1].

    Parameters
    ----------
    iob_true : pd.Series[str]
         Ground truth (correct) IOB2 annotations.
    iob_pred : pd.Series[str]
        Predicted IOB2 annotations.
    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity'
        can span several tokens.
    etypes_map : dict, optional
        Dictionary mapping entity type names in the ground truth annotations
        to the corresponding entity type names in the predicted annotations.
        Useful when entity types have different names in `iob_true` and
        `iob_pred`, e.g. ORGANISM in ground truth and TAXON in predictions.
    return_dict : bool, optional
        If True, return output as dict.

    Returns
    -------
    report : Union[str, OrderedDict]
        Text summary of the precision, recall, F1 score for each entity type.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure

        .. code-block:: python

            {'entity_type 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'entity_type 2': { ... },
              ...
            }

    References
    ----------
    [1] Segura-Bedmar et al. 2013, "Semeval-2013 task 9: Extraction of drug-drug
    interactions from biomedical texts", https://e-archivo.uc3m.es/handle/10016/20455
    """
    _check_consistent_iob(iob_true, iob_pred)

    report = OrderedDict()

    etypes_counts = dict(zip(*unique_etypes(iob_true, mode=mode, return_counts=True)))
    etypes_map = etypes_map or {}
    etypes_map = copy.deepcopy(etypes_map)
    for etype in etypes_counts.keys() - etypes_map.keys():
        etypes_map[etype] = etype

    for etype in etypes_counts.keys():
        if mode == "entity":
            idxs_true = iob2idx(iob_true, etype=etype)
            idxs_pred = iob2idx(iob_pred, etype=etypes_map[etype])
            n_true = len(idxs_true)
            n_pred = len(idxs_pred)
            true_pos = len(idxs_true.merge(idxs_pred, on=["start", "end"], how="inner"))
        elif mode == "token":
            ent_true = iob_true.isin([f"B-{etype}", f"I-{etype}"])
            ent_pred = iob_pred.isin(
                [f"B-{etypes_map[etype]}", f"I-{etypes_map[etype]}"]
            )
            n_true = ent_true.sum().item()
            n_pred = ent_pred.sum().item()
            true_pos = (ent_true & ent_pred).sum().item()
        else:
            raise ValueError(f"Mode {mode} is not available.")

        false_neg = n_true - true_pos
        false_pos = n_pred - true_pos
        precision = true_pos / n_pred if n_pred > 0 else 0
        recall = true_pos / n_true
        f1_score = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
        report[etype] = OrderedDict(
            [
                ("precision", precision),
                ("recall", recall),
                ("f1-score", f1_score),
                ("support", n_true),
            ]
        )

    if return_dict:
        return report
    else:
        out = [
            "".join(
                f"{col_name:>10s}"
                for col_name in ["", "precision", "recall", "f1-score", "support"]
            )
        ]
        for etype, metrics_scores in report.items():
            out.append(
                f"{etype:>10s}"
                + "".join(
                    f"{metrics_scores[metric_name]:>10.2f}"
                    for metric_name in ["precision", "recall", "f1-score"]
                )
                + f"{etypes_counts[etype]:>10d}"
            )
        return "\n".join(out)


def ner_errors(
    iob_true: pd.Series,
    iob_pred: pd.Series,
    tokens: pd.Series,
    mode: str = "entity",
    etypes_map: Optional[dict] = None,
    return_dict: bool = False,
) -> Union[str, OrderedDict]:
    """Build a summary report for the named entity recognition.

    False positives and false negatives for each entity type are collected.
    Evaluation is performed according to the definitions of "errors" from [1].

    Parameters
    ----------
    iob_true :
         Ground truth (correct) IOB2 annotations.
    iob_pred :
        Predicted IOB2 annotations.
    tokens :
        Tokens obtained from tokenization of a text.
    mode :
        Evaluation mode. One of 'entity', 'token': notice that an 'entity'
        can span several tokens.
    etypes_map :
        Dictionary mapping entity type names in the ground truth annotations
        to the corresponding entity type names in the predicted annotations.
        Useful when entity types have different names in `iob_true` and
        `iob_pred`, e.g. ORGANISM in ground truth and TAXON in predictions.
    return_dict :
        If True, return output as dict.

    Returns
    -------
    report : Union[str, OrderedDict]
        Text summary of the precision, recall, F1 score for each entity type.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure

        .. code-block:: python

            {'entity_type 1': {'false_neg': [entity, entity, ...],
                               'false_pos': [entity, entity, ...]},
             'entity_type 2': { ... },
              ...
            }

    References
    ----------
    [1] Segura-Bedmar et al. 2013, "Semeval-2013 task 9: Extraction of drug-drug
    interactions from biomedical texts", https://e-archivo.uc3m.es/handle/10016/20455
    """
    _check_consistent_iob(iob_true, iob_pred)

    if not (len(iob_true) == len(iob_pred) == len(tokens)):
        raise ValueError(
            f"Inputs iob_true (len={len(iob_true)}), iob_pred (len={len(iob_pred)}), "
            f"tokens (len={len(tokens)}) should have equal length."
        )
    etypes = unique_etypes(iob_true)

    etypes_map = etypes_map if etypes_map is not None else {}
    etypes_map = {etype: etypes_map.get(etype, etype) for etype in etypes}

    report = OrderedDict()
    if mode == "entity":
        for etype in etypes:
            idxs_true = iob2idx(iob_true, etype=etype)
            idxs_pred = iob2idx(iob_pred, etype=etypes_map[etype])
            idxs_all = idxs_true.merge(
                idxs_pred, on=["start", "end"], indicator="i", how="outer"
            )
            idxs_false_neg = idxs_all.query('i == "left_only"').drop("i", 1)
            idxs_false_pos = idxs_all.query('i == "right_only"').drop("i", 1)
            report[etype] = {
                "false_neg": sorted(idx2text(tokens, idxs_false_neg).tolist()),
                "false_pos": sorted(idx2text(tokens, idxs_false_pos).tolist()),
            }
    elif mode == "token":
        for etype in etypes:
            etype_symbols_t = [f"B-{etype}", f"I-{etype}"]
            etype_symbols_p = [f"B-{etypes_map[etype]}", f"I-{etypes_map[etype]}"]
            false_neg = tokens.loc[
                iob_true.isin(etype_symbols_t) & (~iob_pred.isin(etype_symbols_p))
            ]
            false_pos = tokens.loc[
                (~iob_true.isin(etype_symbols_t)) & iob_pred.isin(etype_symbols_p)
            ]
            report[etype] = {
                "false_neg": sorted(false_neg.tolist()),
                "false_pos": sorted(false_pos.tolist()),
            }
    else:
        raise ValueError(f"Mode {mode} is not available.")

    if return_dict:
        return report
    else:
        out = []
        for etype, confusion in report.items():
            out.append(f"{etype}")
            out.append("* false negatives")
            for w in confusion["false_neg"]:
                out.append("  - " + w)
            out.append("* false positives")
            for w in confusion["false_pos"]:
                out.append("  - " + w)
            out.append("")
        return "\n".join(out)


def ner_confusion_matrix(
    iob_true: pd.Series,
    iob_pred: pd.Series,
    normalize: Optional[str] = None,
    mode: str = "entity",
) -> pd.DataFrame:
    """Compute confusion matrix to evaluate the accuracy of a NER model.

    Evaluation is performed according to the definitions of "errors" from [1].

    Parameters
    ----------
    iob_true :
         Ground truth (correct) IOB2 annotations.
    iob_pred :
        Predicted IOB2 annotations.
    normalize :
        One of "true", "pred", "all", or None.
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, the confusion matrix will
        not be normalized.
    mode :
        Evaluation mode. One of 'entity', 'token': notice that an 'entity'
        can span several tokens.

    Returns
    -------
    cm : pd.DataFrame
        Dataframe where the index contains the ground truth entity types and
        the columns contain the predicted entity types.

    References
    ----------
    [1] Segura-Bedmar et al. 2013, "Semeval-2013 task 9: Extraction of drug-drug
    interactions from biomedical texts", https://e-archivo.uc3m.es/handle/10016/20455
    """
    _check_consistent_iob(iob_true, iob_pred)

    etypes_true = unique_etypes(iob_true)
    etypes_pred = unique_etypes(iob_pred)

    if mode == "entity":
        cm_vals = np.zeros(
            shape=(len(etypes_true) + 1, len(etypes_pred) + 1), dtype="int64"
        )
        idxs_true = {etype: iob2idx(iob_true, etype=etype) for etype in etypes_true}
        idxs_pred = {etype: iob2idx(iob_pred, etype=etype) for etype in etypes_pred}

        for i, etype_true in enumerate(etypes_true):
            n_true = len(idxs_true[etype_true])
            for j, etype_pred in enumerate(etypes_pred):
                cm_vals[i, j] = len(
                    idxs_true[etype_true].merge(
                        idxs_pred[etype_pred], on=["start", "end"], how="inner"
                    )
                )
            cm_vals[i, -1] = n_true - cm_vals[i, :-1].sum()
        for j, etype_pred in enumerate(etypes_pred):
            cm_vals[-1, j] = len(idxs_pred[etype_pred]) - cm_vals[:-1, j].sum()

        columns = etypes_pred + ["None"]
        index = etypes_true + ["None"]

    elif mode == "token":
        etypes_all = sorted(set(etypes_true + etypes_pred)) + ["O"]

        iob_true = iob_true.str.replace("B-", "").str.replace("I-", "")
        iob_pred = iob_pred.str.replace("B-", "").str.replace("I-", "")
        cm = pd.DataFrame(
            data=sklearn.metrics.confusion_matrix(
                iob_true, iob_pred, labels=etypes_all
            ),
            columns=etypes_all,
            index=etypes_all,
        )
        etypes_true.append("O")
        etypes_pred.append("O")
        cm = cm[etypes_pred].loc[etypes_true]
        cm_vals = cm.values

        columns = etypes_pred[:-1] + ["None"]
        index = etypes_true[:-1] + ["None"]
    else:
        raise ValueError(f"Mode '{mode}' is not available.")

    if normalize == "true":
        cm_vals = cm_vals / cm_vals.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm_vals = cm_vals / cm_vals.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm_vals = cm_vals / cm_vals.sum()
    cm_vals = np.nan_to_num(cm_vals)

    cm = pd.DataFrame(cm_vals, columns=columns, index=index)

    return cm

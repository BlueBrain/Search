"""Collection of functions for evaluation of NER models."""
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from spacy.tokens import Doc


def prodigy2df(cnxn, dataset_name, not_entity_symbol='O'):
    """Convert prodigy annotations to a pd.DataFrame.

    Parameters
    ----------
    cnxn : sqlite3.Connection
        Connection to the prodigy database.
    dataset_name : str
        Name of the dataset from which to retrieve annotations.
    not_entity_symbol : str
        A symbol to use for tokens that are not an entity.

    Returns
    -------
    final_table : pd.DataFrame
        Each row represents one token, the columns are 'source', 'sentence_id', 'class',
        'start_char', end_char', 'id', 'text'.
    """
    first_df = pd.read_sql(f'''
        SELECT *
        FROM example
        WHERE example.id IN
              (
                  SELECT link.example_id
                  FROM link
                  WHERE link.dataset_id IN
                        (
                            SELECT dataset.id
                            FROM dataset
                            WHERE dataset.name = "{dataset_name}"
                        )
              )''',
                           cnxn)

    final_table_rows = []
    for _, row in first_df.iterrows():
        sentence_id = row['id']
        content = json.loads(row['content'])

        if content['answer'] != 'accept':
            continue

        spans = content['spans']  # list of dict

        classes = {}
        for ent in spans:
            for ix, token_ix in enumerate(range(ent['token_start'], ent['token_end'] + 1)):
                ent_label = ent['label'].upper()

                classes[token_ix] = "{}-{}".format('B' if ix == 0 else 'I', ent_label)

        for token in content['tokens']:
            final_table_rows.append({'source': content['meta']['source'],
                                     'sentence_id': sentence_id,
                                     'class': classes.get(token['id'], not_entity_symbol),
                                     'start_char': token['start'],
                                     'end_char': token['end'],
                                     'id': token['id'],
                                     'text': token['text']
                                     })

    final_table = pd.DataFrame(final_table_rows)

    return final_table


def spacy2df(spacy_model, ground_truth_tokenization, not_entity_symbol='O'):
    """Turn NER of a spacy model into a pd.DataFrame.

    Parameters
    ----------
    spacy_model : spacy.language.Language
        Spacy model that will be used for NER (not tokenization).

    ground_truth_tokenization : list
        List of str (words) representing the ground truth tokenization. This will guarantee that the
        ground truth dataframe will be aligned with the prediction dataframe.

    not_entity_symbol : str
        A symbol to use for tokens that are not an entity.

    Returns
    -------
    pd.DataFrame
        Each row represents one token, the columns are 'text' and 'class'.

    Notes
    -----
    One should run the `prodigy2df` first in order to obtain the `ground_truth_tokenization`. If
    it is the case then `ground_truth_tokenization=prodigy_table['text'].to_list()`.
    """
    doc = Doc(spacy_model.vocab, words=ground_truth_tokenization)
    ner = spacy_model.get_pipe('ner')
    new_doc = ner(doc)

    all_rows = []
    for token in new_doc:

        if token.ent_iob_ == "O":
            all_rows.append({'class': not_entity_symbol,
                             'text': token.text,
                             })
        else:
            all_rows.append({'class': "{}-{}".format(token.ent_iob_, token.ent_type_),
                             'text': token.text})

    return pd.DataFrame(all_rows)


def unique_etypes(iob, return_counts=False, mode='entity'):
    """Return the sorted unique entity types from a vector of annotations in IOB format.

    Parameters
    ----------
    iob : pd.Series[str]
        Annotations in the IOB format. Elements of the pd.Series should be either 'O',
        'B-ENTITY_TYPE', or 'I-ENTITY_TYPE', where 'ENTITY_TYPE' is the name of some entity type.

    return_counts : bool, optional
        If True, also return the number of times each unique entity type appears in the input.

    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity' can span several tokens.

    Returns
    -------
    unique : list[str]
        The sorted unique entity types.

    unique_counts : list[int], optional
        The number of times each of the unique entity types comes up in the input. Only provided if
        `return_counts` is True.
    """
    unique = sorted({etype.replace('B-', '').replace('I-', '')
                     for etype in iob.unique() if etype != 'O'})
    if not return_counts:
        return unique
    else:
        if mode == 'entity':
            unique_counts = [np.count_nonzero((iob == f'B-{etype}').values)
                             for etype in unique]
        elif mode == 'token':
            unique_counts = [np.count_nonzero(iob.isin([f'B-{etype}', f'I-{etype}']).values)
                             for etype in unique]
        else:
            raise ValueError(f'Mode \'{mode}\' is not available.')
        return unique, unique_counts


def iob2idx(iob, etype):
    """Retrieve start and end indexes of entities from annotations in IOB format.

    Parameters
    ----------
    iob : pd.Series[str]
        Annotations in the IOB format. Elements of the pd.Series should be either 'O',
        'B-ENTITY_TYPE', or 'I-ENTITY_TYPE', where 'ENTITY_TYPE' is the name of some entity type.

    etype : str
        Name of the entity type of interest.

    Returns
    -------
    idxs : pd.DataFrame[int, int]
        Dataframe with 2 columns, 'start' and 'end', representing start and end position of the
        entities of the specified entity type.
    """
    b_symbol = f'B-{etype}'
    i_symbol = f'I-{etype}'

    iob_next = iob.shift(periods=-1)

    data_dict = {'start': iob.index[iob == b_symbol],
                 'end': iob.index[iob.isin([b_symbol, i_symbol]) &
                                  (iob_next != i_symbol)]}

    idxs = pd.DataFrame(data=data_dict)
    return idxs


def idx2text(tokens, idxs):
    """Retrieve entities text from a list of tokens and start and end indexes.

    Parameters
    ----------
    tokens : pd.Series[str]
        Tokens obtained from tokenization of a text.

    idxs : pd.Series[int, int]
        Dataframe with 2 columns, 'start' and 'end', representing start and end position of the
        entities of the specified entity type.

    Returns
    -------
    texts : pd.Series[str]
        Texts of each entity identified by the indexes provided in input.
    """
    return pd.Series([' '.join(tokens[s:e + 1])
                      for s, e in zip(idxs['start'], idxs['end'])], index=idxs.index, dtype='str')


def ner_report(iob_true, iob_pred, mode='entity', etypes_map=None, return_dict=False):
    """Build a summary report showing the main ner evaluation metrics.

    Parameters
    ----------
    iob_true : pd.Series[str]
         Ground truth (correct) IOB annotations.

    iob_pred : pd.Series[str]
        Predicted IOB annotations.

    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity' can span several tokens.

    etypes_map : dict, optional
        Dictionary mapping entity type names in the ground truth annotations to the corresponding
        entity type names in the predicted annotaitons. Useful when entity types have different
        names in `iob_true` and `iob_pred`, e.g. ORGANISM in ground true and TAXON in predictions.

    return_dict : bool, optional
        If True, return output as dict.

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each entity type.
        Dictionary returned if output_dict is True. Dictionary has the following structure::
            {'entity_type 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'entity_type 2': { ... },
              ...
            }
    """
    report = OrderedDict()

    etypes_counts = dict(zip(*unique_etypes(iob_true, mode=mode, return_counts=True)))
    etypes_map = etypes_map if etypes_map is not None else dict()
    etypes_map = {etype: etypes_map.get(etype, etype)
                  for etype in etypes_counts.keys()}

    for etype in etypes_counts.keys():
        if mode == 'entity':
            idxs_true = iob2idx(iob_true, etype=etype)
            idxs_pred = iob2idx(iob_pred, etype=etypes_map[etype])
            n_true = len(idxs_true)
            n_pred = len(idxs_pred)
            true_pos = np.count_nonzero((idxs_true['start'].isin(idxs_pred['start']) &
                                         idxs_true['end'].isin(idxs_pred['end'])).values)
        elif mode == 'token':
            ent_true = iob_true.isin([f'B-{etype}', f'I-{etype}'])
            ent_pred = iob_pred.isin([f'B-{etypes_map[etype]}', f'I-{etypes_map[etype]}'])
            n_true = np.count_nonzero(ent_true.values)
            n_pred = np.count_nonzero(ent_pred.values)
            true_pos = np.count_nonzero((ent_true & ent_pred).values)
        else:
            raise ValueError(f'Mode {mode} is not available.')

        false_neg = n_true - true_pos
        false_pos = n_pred - true_pos
        precision = true_pos / n_pred if n_pred > 0 else 0
        recall = true_pos / n_true
        f1_score = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
        report[etype] = OrderedDict([('precision', precision),
                                     ('recall', recall),
                                     ('f1-score', f1_score),
                                     ('support', n_true)])

    if return_dict:
        return report
    else:
        out = [''.join(f'{col_name:>10s}'
                       for col_name in ['', 'precision', 'recall', 'f1-score', 'support'])]
        for etype, metrics_scores in report.items():
            out.append(f'{etype:>10s}'
                       + ''.join(f'{metrics_scores[metric_name]:>10.2f}'
                                 for metric_name in ['precision', 'recall', 'f1-score'])
                       + f'{etypes_counts[etype]:>10d}')
        return '\n'.join(out)


def ner_errors(iob_true, iob_pred, tokens, mode='entity', etypes_map=None, return_dict=False):
    """Build a summary report collecting false positives and false negatives for each entity type.

    Parameters
    ----------
    iob_true : pd.Series[str]
         Ground truth (correct) IOB annotations.

    iob_pred : pd.Series[str]
        Predicted IOB annotations.

    tokens : pd.Series[str]
        Tokens obtained from tokenization of a text.

    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity' can span several tokens.

    etypes_map : dict, optional
        Dictionary mapping entity type names in the ground truth annotations to the corresponding
        entity type names in the predicted annotaitons. Useful when entity types have different
        names in `iob_true` and `iob_pred`, e.g. ORGANISM in ground true and TAXON in predictions.

    return_dict : bool, optional
        If True, return output as dict.

    Returns
    -------
    report : string / dict
        Text summary of the precision, recall, F1 score for each entity type.
        Dictionary returned if output_dict is True. Dictionary has the following structure::
            {'entity_type 1': {'false_neg': [entity, entity, ...],
                         'false_pos': [entity, entity, ...]},
             'entity_type 2': { ... },
              ...
            }
    """
    assert len(iob_true) == len(iob_pred) == len(tokens)
    etypes = unique_etypes(iob_true)

    etypes_map = etypes_map if etypes_map is not None else dict()
    etypes_map = {etype: etypes_map.get(etype, etype)
                  for etype in etypes}

    report = OrderedDict()
    if mode == 'entity':
        for etype in etypes:
            idxs_true = iob2idx(iob_true, etype=etype)
            idxs_pred = iob2idx(iob_pred, etype=etypes_map[etype])
            idxs_false_neg = idxs_true[(~idxs_true['start'].isin(idxs_pred['start'])) |
                                       (~idxs_true['end'].isin(idxs_pred['end']))]
            idxs_false_pos = idxs_pred[(~idxs_pred['start'].isin(idxs_true['start'])) |
                                       (~idxs_pred['end'].isin(idxs_true['end']))]
            report[etype] = ({'false_neg': sorted(idx2text(tokens, idxs_false_neg).tolist()),
                              'false_pos': sorted(idx2text(tokens, idxs_false_pos).tolist())})
    elif mode == 'token':
        for etype in etypes:
            etype_symbols = [f'B-{etype}', f'I-{etype}']
            false_neg = tokens[iob_true.isin(etype_symbols) & (~iob_pred.isin(etype_symbols))]
            false_pos = tokens[(~iob_true.isin(etype_symbols)) & iob_pred.isin(etype_symbols)]
            report[etype] = ({'false_neg': sorted(false_neg.tolist()),
                              'false_pos': sorted(false_pos.tolist())})
    else:
        raise ValueError(f'Mode {mode} is not available.')

    if return_dict:
        return report
    else:
        out = []
        for etype, confusion in report.items():
            out.append(f'{etype}')
            out.append('* false negatives')
            for w in confusion['false_neg']:
                out.append('  - ' + w)
            out.append('* false positives')
            for w in confusion['false_pos']:
                out.append('  - ' + w)
            out.append('')
        return '\n'.join(out)


def confusion_matrix(iob_true, iob_pred, normalize=None, mode='entity'):
    """Compute confusion matrix to evaluate the accuracy of a NER model.

    Parameters
    ----------
    iob_true : pd.Series[str]
         Ground truth (correct) IOB annotations.

    iob_pred : pd.Series[str]
        Predicted IOB annotations.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the
        population. If None, the confusion matrix will not be normalized.

    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity' can span several tokens.

    Returns
    -------
    cm : pd.DataFrame
        Dataframe where the index contains the ground truth entity types and the columns contain the
        predicted entity types.
    """
    etypes_true = unique_etypes(iob_true)
    etypes_pred = unique_etypes(iob_pred)

    if mode == 'entity':
        cm_vals = np.zeros(shape=(len(etypes_true) + 1, len(etypes_pred) + 1))
        idxs_true = {etype: iob2idx(iob_true, etype=etype) for etype in etypes_true}
        idxs_pred = {etype: iob2idx(iob_pred, etype=etype) for etype in etypes_pred}

        for i, etype_true in enumerate(etypes_true):
            n_true = len(idxs_true[etypes_true])
            for j, etype_pred in enumerate(etypes_pred):
                idxs_pred = iob2idx(iob_pred, etype=etype_pred)
                cm_vals[i, j] = np.count_nonzero(
                    (idxs_true[etype_true]['start'].isin(idxs_pred[etype_pred]['start']) &
                     idxs_true[etype_pred]['end'].isin(idxs_pred[etype_pred]['end'])
                     ).values)
            cm_vals[i, -i] = n_true - cm_vals[i, :-1].sum()
        for j, etype_pred in enumerate(etypes_pred):
            cm_vals[-1, j] = len(idxs_pred[etype_pred]) - cm_vals[:-1, j]

    elif mode == 'token':
        etypes_all = sorted(set(etypes_true + etypes_pred)) + ['O']

        iob_true = iob_true.str.replace('B-', '').str.replace('I-', '')
        iob_pred = iob_pred.str.replace('B-', '').str.replace('I-', '')
        cm = pd.DataFrame(
            data=sklearn.metrics.confusion_matrix(iob_true, iob_pred, labels=etypes_all),
            columns=etypes_all,
            index=etypes_all
        )
        etypes_true.append('O')
        etypes_pred.append('O')
        cm = cm[etypes_pred].loc[etypes_true]
        cm_vals = cm.values
    else:
        raise ValueError(f'Mode \'{mode}\' is not available.')

    if normalize == 'true':
        cm_vals = cm_vals / cm_vals.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm_vals = cm_vals / cm_vals.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm_vals = cm_vals / cm_vals.sum()
    cm_vals = np.nan_to_num(cm_vals)

    cm = pd.DataFrame(cm_vals, columns=etypes_pred[:-1] + ['None'], index=etypes_true[:-1] + ['None'])

    return cm


def plot_confusion_matrix(iob_true, iob_pred, normalize=None, mode='entity', cmap='viridis',
                          ax=None):
    """Plot Confusion Matrix for NER evaluation.

    Parameters
    ----------
    iob_true : pd.Series[str]
         Ground truth (correct) IOB annotations.

    iob_pred : pd.Series[str]
        Predicted IOB annotations.

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the
        population. If None, the confusion matrix will not be normalized.

    mode : str, optional
        Evaluation mode. One of 'entity', 'token': notice that an 'entity' can span several tokens.

    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.

    ax : matplotlib Axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created and returned.

    Returns
    -------
    ax : matplotlib Axes
        Axes object where the matrix was plotted.
    """
    cm = confusion_matrix(iob_true, iob_pred, normalize=normalize, mode=mode)

    x = cm.values
    e_types_true = cm.index
    e_types_pred = cm.columns

    if ax is None:
        f, ax = plt.subplots(figsize=(len(e_types_pred) * .6, len(e_types_true) * .6))

    max_val = 1 if (normalize is not None) else x[:-1, :-1].max()
    ax.imshow(-np.minimum(x, max_val), cmap=cmap, vmin=-max_val, vmax=0)

    for i in range(len(x)):
        for j in range(len(x.T)):
            c = 'k' if x[i, j] < max_val / 2 else 'w'
            text = f'{round(100 * x[i, j]):d}%' if (normalize is not None) else f'{x[i, j]:,d}'
            ax.text(j, i, text, ha='center', va='center', color=c, fontsize=12)

    ax.set_xticks(np.arange(len(e_types_pred)))
    ax.set_yticks(np.arange(len(e_types_true)))
    ax.set_xticklabels(list(e_types_pred), rotation=90, fontsize=15)
    ax.set_yticklabels(e_types_true, fontsize=15)
    ax.set_xlabel('Predicted', fontsize=20)
    ax.set_ylabel('True', fontsize=20)

    return ax

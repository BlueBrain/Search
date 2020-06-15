import json
import pandas as pd

from spacy.tokens import Doc


# helpers
def prodigy2df(cnxn):
    """Convert prodigy annotations to a pd.DataFrame.

    Parameters
    ----------
    cursor : sqlite3.Cursor
        Cursor to the prodigy database.

    Returns
    -------
    pd.DataFrame
        Each row represents one token, the columns are 'source', 'sentence_id',  'class',
        'start_char', end_char', 'token_id', 'text'.
    """
    not_entity_symbol = 'O'

    first_df = pd.read_sql('SELECT * FROM example', cnxn)

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
                if ent_label == 'ORGANS':
                    ent_label = 'ORGAN'

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


# helpers
def spacy2df(spacy_model, ground_truth_tokenization):
    """Turn NER of a spacy model into a pd.DataFrame

    Paramters
    ---------
    spacy_model : spacy.language.Language
        Spacy model that will be used for NER (not tokenization)

    ground_truth_tokenization : list
        List of str (words) representing the ground truth tokenization. This will guarantee
        that the ground truth dataframe will be aligned with the prediction dataframe.

    """
    not_entity_symbol = 'O'

    doc = Doc(spacy_model.vocab, words=ground_truth_tokenization['text'].to_list())
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

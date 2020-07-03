import json
import sqlite3
from unittest.mock import Mock

import pandas as pd
import pytest

from bbsearch.mining import prodigy2df, spacy2df
from bbsearch.mining.eval import unique_etypes, iob2idx, idx2text


class TestProdigy2df:

    @pytest.mark.parametrize('answer', ['accept', 'ignore'])
    def test_overall(self, monkeypatch, answer):
        prodigy_content = {
            'answer': answer,
            'meta': {'pattern': '', 'source': 'amazing source'},

            'spans': [{'label': 'PERSON', 'start': 0, 'end': 14, 'token_start': 0, 'token_end': 1},
                      {'label': 'GPE', 'start': 32, 'end': 38, 'token_start': 6, 'token_end': 6},
                      {'label': 'DATE', 'start': 39, 'end': 48, 'token_start': 7, 'token_end': 7}],

            'tokens': [{'text': 'Britney', 'start': 0, 'end': 7, 'id': 0},
                       {'text': 'Spears', 'start': 8, 'end': 14, 'id': 1},
                       {'text': 'had', 'start': 15, 'end': 18, 'id': 2},
                       {'text': 'a', 'start': 19, 'end': 20, 'id': 3},
                       {'text': 'concert', 'start': 21, 'end': 28, 'id': 4},
                       {'text': 'in', 'start': 29, 'end': 31, 'id': 5},
                       {'text': 'Brazil', 'start': 32, 'end': 38, 'id': 6},
                       {'text': 'yesterday', 'start': 39, 'end': 48, 'id': 7},
                       {'text': '.', 'start': 48, 'end': 49, 'id': 8}]
        }

        n_examples = 2
        examples_table = pd.DataFrame([{'id': i, 'content': json.dumps(prodigy_content)}
                                       for i in range(n_examples)])

        fake_read_sql = Mock()
        fake_read_sql.return_value = examples_table
        fake_cnxn = Mock(spec=sqlite3.Connection)

        monkeypatch.setattr('bbsearch.mining.eval.pd.read_sql', fake_read_sql)

        df = prodigy2df(fake_cnxn, dataset_name='cord19_JohnSmith')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_examples * (len(prodigy_content['tokens']) if answer == 'accept' else 0)

        if answer == 'accept':
            assert {'source', 'sentence_id', 'class', 'start_char', 'end_char', 'id',
                    'text'} == set(df.columns)


class TestSpacy2df:
    def test_overall(self, model_entities):
        ground_truth_tokenization = ['Britney', 'Spears', 'had', 'a', 'concert', 'in', 'Brazil',
                                     'yesterday', '.']
        classes = ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'B-GPE', 'B-DATE', 'O']

        df = spacy2df(model_entities, ground_truth_tokenization)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(ground_truth_tokenization)
        assert df['class'].to_list() == classes
        assert 'class' in df.columns
        assert 'text' in df.columns


@pytest.mark.parametrize('annotations, etypes, counts', [
    ('annotations_1',
     ['CONDITION', 'DISEASE', 'ORGANISM', 'PATHWAY'],
     {'iob': [2, 4, 6, 2], 'token': [3, 9, 9, 4]}
     ),
    ('annotations_2',
     ['CONDITION', 'DISEASE', 'PATHWAY', 'TAXON'],
     {'iob': [1, 6, 1, 8], 'token': [1, 11, 2, 11]}
     )
])
def test_unique_etypes(ner_annotations, annotations, etypes, counts):
    for mode in ('iob', 'token'):
        assert unique_etypes(ner_annotations[annotations], return_counts=False, mode=mode) \
            == etypes
        assert unique_etypes(ner_annotations[annotations], return_counts=True, mode=mode) \
            == (etypes, counts[mode])


@pytest.mark.parametrize('annotations, etype, idxs', [
    ('annotations_1', 'CONDITION', [[103, 104], [108, 108]]),
    ('annotations_1', 'DISEASE', [[34, 37], [40, 40], [120, 121], [148, 149]]),
    ('annotations_2', 'PATHWAY', [[135, 136]]),
    ('annotations_1', 'POTATOES', None)
])
def test_iob2idx(ner_annotations, annotations, etype, idxs):
    if idxs is not None:
        pd.testing.assert_frame_equal(iob2idx(ner_annotations[annotations], etype),
                                      pd.DataFrame(data=idxs, columns=['start', 'end']))
    else:
        pd.testing.assert_frame_equal(iob2idx(ner_annotations[annotations], etype),
                                      pd.DataFrame(data={'start': [], 'end': []},
                                                   index=pd.Int64Index([]),
                                                   dtype='int64'))


@pytest.mark.parametrize('annotations, etype, texts', [
    ('annotations_1', 'CONDITION', ['worldwide outbreak', 'hospitalization']),
    ('annotations_1', 'ORGANISM', ['human coronaviruses', 'Human coronaviruses',
                                   'Human coronaviruses', 'HCoVs', 'infant', 'infant']),
    ('annotations_2', 'PATHWAY', ['respiratory immunity']),
    ('annotations_1', 'POTATOES', [])])
def test_idx2text(ner_annotations, annotations, etype, texts):
    assert texts == \
           idx2text(ner_annotations.text, iob2idx(ner_annotations[annotations], etype)).tolist()

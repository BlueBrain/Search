import json
import sqlite3
from unittest.mock import Mock

import pandas as pd
import pytest

from bbsearch.mining import prodigy2df, spacy2df


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

        df = prodigy2df(fake_cnxn)

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

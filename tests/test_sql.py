"""Collection of tests regarding the Database creation. """
from importlib import import_module
import inspect
import pytest

import pandas as pd

from bbsearch.sql import find_paragraph, retrieve_sentences_from_sentence_id


class TestNoSQL:

    @pytest.mark.parametrize('module_name', ['article_saver', 'embedding_models',
                                             # 'entrypoints.database_entrypoint',
                                             # 'entrypoints.embedding_server_entrypoint',
                                             # 'entrypoints.embeddings_entrypoint',
                                             # 'entrypoints.mining_server_entrypoint',
                                             # 'entrypoints.search_server_entrypoint',
                                             'mining.attributes',
                                             'mining.eval',
                                             'mining.pipeline',
                                             'mining.relation',
                                             'remote_searcher',
                                             'search',
                                             'server.embedding_server',
                                             'server.mining_server',
                                             'server.search_server',
                                             'utils',
                                             'widget'])
    def test_sql_queries(self, module_name):
        module = import_module(f'bbsearch.{module_name}')
        source_code = inspect.getsource(module)
        assert 'SELECT' not in source_code


class TestSQLQueries:

    @pytest.mark.parametrize('sentence_id', [[7], [7, 9], [-1], [9, 9]])
    def test_retrieve_sentence_from_sentence_id(self, sentence_id, fake_sqlalchemy_engine):
        """Test that the find paragraph method is working."""
        sentence_text = retrieve_sentences_from_sentence_id(sentence_id=sentence_id,
                                                            engine=fake_sqlalchemy_engine)
        assert isinstance(sentence_text, pd.DataFrame)
        if sentence_id == [-1]:
            assert sentence_text.shape[0] == 0
        else:
            assert sentence_text.shape[0] == len(set(sentence_id))

    # def test_retrieve_sentences_from_section_name(self):

    def test_find_paragraph(self, fake_sqlalchemy_engine):
        """Test that the find paragraph method is working."""
        sentence_id = 7
        sentence_text = pd.read_sql(f'SELECT text FROM sentences WHERE sentence_id = {sentence_id}',
                                    fake_sqlalchemy_engine).iloc[0]['text']
        paragraph_text = find_paragraph(sentence_id, fake_sqlalchemy_engine)
        assert sentence_text in paragraph_text

"""Collection of tests regarding the Database creation. """
from importlib import import_module
import inspect
import pytest

from bbsearch.sql import find_paragraph


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

    def test_find_paragraph(self, fake_db_cursor, fake_db_cnxn, fake_sqlalchemy_cnxn):
        """Test that the find paragraph method is working."""
        sentence_id = 7
        sentence_text = fake_db_cursor.execute('SELECT text FROM sentences WHERE sentence_id = ?',
                                               [sentence_id]).fetchone()[0]
        paragraph_text = find_paragraph(sentence_id, fake_db_cnxn)
        assert sentence_text in paragraph_text
        paragraph_text = find_paragraph(sentence_id, fake_sqlalchemy_cnxn)
        assert sentence_text in paragraph_text

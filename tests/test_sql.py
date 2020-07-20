"""Collection of tests regarding the Database creation. """
from importlib import import_module
import inspect
import pytest

import numpy as np
import pandas as pd

from bbsearch.sql import find_paragraph, retrieve_sentences_from_sentence_id, \
    retrieve_sentences_from_section_name, retrieve_article_metadata, retrieve_article, \
    retrieve_paragraph


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
        """Test that retrieve sentences from sentence_id is working."""
        sentence_text = retrieve_sentences_from_sentence_id(sentence_id=sentence_id,
                                                            engine=fake_sqlalchemy_engine)
        assert isinstance(sentence_text, pd.DataFrame)
        if sentence_id == [-1]:
            assert sentence_text.shape[0] == 0
        else:
            assert sentence_text.shape[0] == len(set(sentence_id))
        assert np.all(sentence_text.columns == ['sentence_id', 'text'])

    def test_from_section_name(self, fake_sqlalchemy_engine, test_parameters):
        """Test that retrieve sentences from section_name is working."""
        section_name = ['section_1']
        sentence_text = retrieve_sentences_from_section_name(section_name=section_name,
                                                             engine=fake_sqlalchemy_engine)
        assert isinstance(sentence_text, pd.DataFrame)
        n_article = pd.read_sql('SELECT COUNT(DISTINCT(article_id)) FROM articles',
                                fake_sqlalchemy_engine).iloc[0, 0]
        number_of_rows = n_article * len(set(section_name)) * test_parameters['n_sentences_per_section']
        assert sentence_text.shape[0] == number_of_rows
        assert np.all(sentence_text.columns == ['sentence_id', 'section_name', 'text'])

    def test_article_metadata(self, fake_sqlalchemy_engine):
        """Test that retrieve article metadata from sentence_id is working."""
        sentence_id = 1
        article = retrieve_article_metadata(sentence_id=sentence_id,
                                            engine=fake_sqlalchemy_engine)
        assert isinstance(article, pd.DataFrame)
        assert article.shape[0] == 1
        assert np.all(article.columns == ['article_id', 'cord_uid', 'sha', 'source_x', 'title',
                                          'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
                                          'publish_time', 'authors', 'journal', 'mag_id',
                                          'who_covidence_id', 'arxiv_id', 'pdf_json_files',
                                          'pmc_json_files', 'url', 's2_id'])

    def test_retrieve_article(self, fake_sqlalchemy_engine):
        """Test that retrieve article text from sentence_id is working."""
        sentence_id = 1
        article = retrieve_article(sentence_id=sentence_id,
                                   engine=fake_sqlalchemy_engine)
        assert isinstance(article, str)

    def test_retrieve_paragraph(self, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from sentence_id is working."""
        sentence_id = 1
        paragraph = retrieve_paragraph(sentence_id=sentence_id,
                                       engine=fake_sqlalchemy_engine)
        assert isinstance(paragraph, str)

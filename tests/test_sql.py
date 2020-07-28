"""Collection of tests regarding the Database creation. """
from importlib import import_module
import inspect
import pytest

import numpy as np
import pandas as pd

from bbsearch.sql import retrieve_sentences_from_sentence_id, \
    retrieve_sentences_from_section_name, retrieve_article_metadata_from_sentence_id, \
    retrieve_article, retrieve_paragraph, retrieve_paragraph_from_sentence_id, \
    retrieve_article_metadata_from_article_id


class TestNoSQL:

    @pytest.mark.parametrize('module_name', ['embedding_models',
                                             'mining.attributes',
                                             'mining.pipeline',
                                             'mining.relation',
                                             'remote_searcher',
                                             'search',
                                             'server.embedding_server',
                                             'server.mining_server',
                                             'server.search_server',
                                             'utils',
                                             'widgets.article_saver',
                                             'widgets.mining_widget',
                                             'widgets.search_widget'])
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
        assert np.all(sentence_text.columns == \
                      ['article_id', 'sentence_id', 'section_name', 'text',
                       'paragraph_pos_in_article'])

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
        article = retrieve_article_metadata_from_sentence_id(sentence_id=sentence_id,
                                                             engine=fake_sqlalchemy_engine)
        assert isinstance(article, pd.DataFrame)
        assert article.shape[0] == 1
        assert np.all(article.columns == ['article_id', 'cord_uid', 'sha', 'source_x', 'title',
                                          'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
                                          'publish_time', 'authors', 'journal', 'mag_id',
                                          'who_covidence_id', 'arxiv_id', 'pdf_json_files',
                                          'pmc_json_files', 'url', 's2_id'])

        article_id = 1
        article = retrieve_article_metadata_from_article_id(article_id=article_id,
                                                            engine=fake_sqlalchemy_engine)
        assert isinstance(article, pd.DataFrame)
        assert article.shape[0] == 1
        assert np.all(article.columns == ['article_id', 'cord_uid', 'sha', 'source_x', 'title',
                                          'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
                                          'publish_time', 'authors', 'journal', 'mag_id',
                                          'who_covidence_id', 'arxiv_id', 'pdf_json_files',
                                          'pmc_json_files', 'url', 's2_id'])

    def test_retrieve_article(self, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from sentence_id is working."""
        article_id = 1
        article = retrieve_article(article_id=(article_id, ),
                                   engine=fake_sqlalchemy_engine)
        assert isinstance(article, pd.DataFrame)

    def test_retrieve_paragraph_from_sentence_id(self, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from sentence_id is working."""
        sentence_id = 1
        paragraph = retrieve_paragraph_from_sentence_id(sentence_id=sentence_id,
                                                        engine=fake_sqlalchemy_engine)
        sentence_text = retrieve_sentences_from_sentence_id(sentence_id=(sentence_id, ),
                                                            engine=fake_sqlalchemy_engine)
        assert isinstance(paragraph, str)
        assert sentence_text['text'].iloc[0] in paragraph

    def test_retrieve_paragraph(self, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from sentence_id is working."""
        article_id, paragraph_pos = 1, 0
        paragraph = retrieve_paragraph(identifier=(article_id, paragraph_pos),
                                       engine=fake_sqlalchemy_engine)
        assert isinstance(paragraph, pd.DataFrame)
        assert np.all(paragraph.columns == ['article_id', 'text',
                                            'section_name', 'paragraph_pos_in_article'])
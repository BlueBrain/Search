"""Collection of tests regarding the Database creation. """
from importlib import import_module
import inspect
import pytest

import numpy as np
import pandas as pd

from bbsearch.sql import (get_sentence_ids_by_condition, retrieve_sentences_from_sentence_ids,
                          retrieve_articles, retrieve_paragraph, retrieve_paragraph_from_sentence_id,
                          retrieve_article_metadata_from_article_id)


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
    def test_retrieve_sentence_from_sentence_ids(self, sentence_id, fake_sqlalchemy_engine):
        """Test that retrieve sentences from sentence_id is working."""
        sentence_text = retrieve_sentences_from_sentence_ids(sentence_ids=sentence_id,
                                                             engine=fake_sqlalchemy_engine)
        assert isinstance(sentence_text, pd.DataFrame)
        if sentence_id == [-1]:  # invalid sentence_id
            assert sentence_text.shape[0] == 0
        else:
            assert sentence_text.shape[0] == len(set(sentence_id))
            assert set(sentence_text['sentence_id'].to_list()) == set(sentence_id)
        assert np.all(sentence_text.columns ==
                      ['article_id', 'sentence_id', 'section_name', 'text',
                       'paragraph_pos_in_article'])

    @pytest.mark.parametrize('sentence_id', [1, 2, 3, 0, -100])
    def test_retrieve_paragraph_from_sentence_id(self, sentence_id, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from sentence_id is working."""
        paragraph = retrieve_paragraph_from_sentence_id(sentence_id=sentence_id,
                                                        engine=fake_sqlalchemy_engine)
        sentence_text = retrieve_sentences_from_sentence_ids(sentence_ids=(sentence_id,),
                                                             engine=fake_sqlalchemy_engine)
        if sentence_id == 0 or sentence_id == -100:  # invalid sentence_id
            assert paragraph is None
        else:
            assert isinstance(paragraph, str)
            assert sentence_text['text'].iloc[0] in paragraph

    @pytest.mark.parametrize('identifier', [(1, 0), (-2, 0), (1, -100)])
    def test_retrieve_paragraph(self, identifier, fake_sqlalchemy_engine):
        """Test that retrieve paragraph text from identifier is working."""
        article_id, paragraph_pos_in_article = identifier
        paragraph = retrieve_paragraph(article_id,
                                       paragraph_pos_in_article,
                                       engine=fake_sqlalchemy_engine)
        assert isinstance(paragraph, pd.DataFrame)
        assert np.all(paragraph.columns == ['article_id', 'text',
                                            'section_name', 'paragraph_pos_in_article'])
        if identifier == (1, 0):  # valid identifier
            assert paragraph.shape == (1, 4)
        else:
            assert len(paragraph.index) == 0

    @pytest.mark.parametrize('article_id', [1, 2, -2, -100])
    def test_article_metadata(self, article_id, fake_sqlalchemy_engine):
        """Test that retrieve article metadata from article_id is working."""
        article = retrieve_article_metadata_from_article_id(article_id=article_id,
                                                            engine=fake_sqlalchemy_engine)
        assert isinstance(article, pd.DataFrame)
        assert np.all(article.columns == ['article_id', 'cord_uid', 'sha', 'source_x', 'title',
                                          'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
                                          'publish_time', 'authors', 'journal', 'mag_id',
                                          'who_covidence_id', 'arxiv_id', 'pdf_json_files',
                                          'pmc_json_files', 'url', 's2_id'])
        if article_id >= 0:  # valid article_id for the fake_sqlalchemy_engine (>0 for the real one)
            assert len(article.index) == 1
        else:
            assert len(article.index) == 0

    @pytest.mark.parametrize('article_id', [[1], [1, 2], [0], [-100]])
    def test_retrieve_article(self, article_id, fake_sqlalchemy_engine, test_parameters):
        """Test that retrieve article from article_id is working."""
        articles = retrieve_articles(article_ids=article_id,
                                     engine=fake_sqlalchemy_engine)
        assert isinstance(articles, pd.DataFrame)
        if min(article_id) >= 0:  # valid article_id
            assert set(articles['article_id'].to_list()) == set(article_id)
            assert articles.shape[0] == len(set(article_id)) * \
                   test_parameters['n_sections_per_article']

    @pytest.mark.parametrize('sentence_ids', [[1, 2, 5], None])
    @pytest.mark.parametrize('conditions', [[], ['1']])
    def test_get_sentence_ids_by_condition(self, fake_sqlalchemy_engine, sentence_ids, conditions):

        n_sentences = pd.read_sql('SELECT COUNT(*) FROM sentences',
                                  fake_sqlalchemy_engine).iloc[0, 0]

        retrieved_sentences = get_sentence_ids_by_condition(conditions,
                                                            fake_sqlalchemy_engine,
                                                            sentence_ids=sentence_ids)

        expected_length = len(sentence_ids) if sentence_ids is not None else n_sentences

        assert isinstance(retrieved_sentences, list)
        assert len(retrieved_sentences) == expected_length

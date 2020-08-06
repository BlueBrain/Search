"""Collection of tests regarding the Database creation. """
import inspect
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from bbsearch.sql import (
    SentenceFilter,
    retrieve_article_metadata_from_article_id,
    retrieve_articles,
    retrieve_paragraph,
    retrieve_paragraph_from_sentence_id,
    retrieve_sentences_from_sentence_ids,
)


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

    # @pytest.mark.parametrize('sentence_ids', [[1, 2, 5], None])
    # @pytest.mark.parametrize('conditions', [[], ['1']])
    # def test_get_sentence_ids_by_condition(self, fake_sqlalchemy_engine, sentence_ids, conditions):
    #
    #     n_sentences = pd.read_sql('SELECT COUNT(*) FROM sentences',
    #                               fake_sqlalchemy_engine).iloc[0, 0]
    #
    #     retrieved_sentences = get_sentence_ids_by_condition(conditions,
    #                                                         fake_sqlalchemy_engine,
    #                                                         sentence_ids=sentence_ids)
    #
    #     expected_length = len(sentence_ids) if sentence_ids is not None else n_sentences
    #
    #     assert isinstance(retrieved_sentences, list)
    #     assert len(retrieved_sentences) == expected_length


class TestSentenceFilter:
    @pytest.mark.parametrize("has_journal", [True, False])
    def test_no_filter(self, fake_sqlalchemy_engine, has_journal):
        all_ids = pd.read_sql(
            "SELECT sentence_id FROM sentences",
            fake_sqlalchemy_engine)["sentence_id"]
        no_filter_ids = SentenceFilter(fake_sqlalchemy_engine).run()

        assert len(all_ids) == len(no_filter_ids)
        assert set(all_ids) == set(no_filter_ids)

    @pytest.mark.parametrize("has_journal", [True, False])
    @pytest.mark.parametrize("indices", [[], [1], [1, 2, 3]])
    @pytest.mark.parametrize("date_range", [None, (1960, 2010), (0, 0)])
    @pytest.mark.parametrize("exclusion_text", ["", "virus", "VIRus\n\nease"])
    def test_sentence_filter(
            self,
            fake_sqlalchemy_engine,
            has_journal,
            indices,
            date_range,
            exclusion_text
    ):
        # Recreate filtering in pandas for comparison
        df_all_articles = pd.read_sql("SELECT * FROM articles", fake_sqlalchemy_engine)
        df_all_sentences = pd.read_sql("SELECT * FROM sentences", fake_sqlalchemy_engine)

        # has journal
        if has_journal:
            df_all_articles = df_all_articles[~df_all_articles["journal"].isna()]
        # date range
        if date_range is not None:
            year_from, year_to = date_range
            all_dates = pd.to_datetime(df_all_articles["publish_time"])
            df_all_articles = df_all_articles[all_dates.dt.year.between(year_from, year_to)]
        # selected sentences that correspond to filtered articles
        df_all_sentences = df_all_sentences[
            df_all_sentences["article_id"].isin(df_all_articles["article_id"])]
        # indices
        df_all_sentences = df_all_sentences[
            df_all_sentences["sentence_id"].isin(indices)]
        # text exclusions
        exclusion_strings = exclusion_text.split()
        exclusion_strings = map(lambda s: s.lower(), exclusion_strings)
        exclusion_strings = filter(lambda s: len(s) > 0, exclusion_strings)
        pattern = "|".join(exclusion_strings)
        if len(pattern) > 0:
            df_all_sentences = df_all_sentences[~df_all_sentences["text"].str.contains(pattern)]
        ids_from_pandas = df_all_sentences["sentence_id"].tolist()

        # Construct filter with various conditions
        sentence_filter = (
            SentenceFilter(fake_sqlalchemy_engine)
            .only_with_journal(has_journal)
            .restrict_sentences_ids_to(indices)
            .date_range(date_range)
            .exclude_strings(exclusion_text.split())
        )

        # Get filtered ids in a single run
        ids_from_run = sentence_filter.run()

        # Get filtered IDs by iteration
        ids_from_iterate = []
        for ids in sentence_filter.iterate(chunk_size=4):
            assert len(ids) <= 4
            ids_from_iterate += list(ids)

        # Running and iterating should give the same results
        assert len(ids_from_run) == len(ids_from_iterate) == len(ids_from_pandas)
        assert set(ids_from_run) == set(ids_from_iterate) == set(ids_from_pandas)

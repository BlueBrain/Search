import pandas as pd

from bbsearch.article_saver import ArticleSaver
from bbsearch.widget import SAVING_OPTIONS


class TestArticleSaver:

    def test_article_saver(self, fake_db_cursor):
        """Test that article_saver is good. """
        article_saver = ArticleSaver(database=fake_db_cursor.connection)

        # Check the possible article_id, paragraphs_id of the fake database
        # Create a fake article_saver.saved_articles dictionary
        # (Which should be the output of the widget)
        results = fake_db_cursor.execute(
            """SELECT sha, article_id FROM article_id_2_sha
            WHERE sha is NOT NULL""").fetchall()
        all_articles_paragraphs_id = dict()
        for sha, article_id in results:
            all_paragraphs_id = fake_db_cursor.execute(
                """SELECT paragraph_id FROM paragraphs
                WHERE sha is ?""", [sha]).fetchall()
            all_articles_paragraphs_id[article_id] = [paragraph_id for (paragraph_id,) in all_paragraphs_id]
            # For all articles extract only the first of their paragraphs
            article_saver.saved_articles[article_id,
                                         all_articles_paragraphs_id[article_id][0]] = SAVING_OPTIONS['paragraph']

        # For the last article extract all its paragraphs
        article_saver.saved_articles[article_id,
                                     all_articles_paragraphs_id[article_id][0]] = SAVING_OPTIONS['article']
        n_paragraphs_full_article = len(all_paragraphs_id)

        # Check that the retrieving of the different text is working
        article_saver.retrieve_text()
        assert isinstance(article_saver.df_chosen_texts, pd.DataFrame)
        assert article_saver.df_chosen_texts.columns.to_list() == ['article_id', 'section_name', 'paragraph_id', 'text']
        assert len(article_saver.df_chosen_texts) == len(all_articles_paragraphs_id) + n_paragraphs_full_article - 1

        # Check summary table
        summary_table = article_saver.summary_table()
        assert isinstance(summary_table, pd.DataFrame)

        # Check that the status feedback is working fine
        for article_id, paragraph_id_list in all_articles_paragraphs_id.items():
            assert isinstance(article_saver.status_on_article_retrieve((article_id, paragraph_id_list[0])), str)
        assert isinstance(article_saver.status_on_article_retrieve((article_id, '1000')), str)

        ARTICLE_ID = 'w8579f54'
        # Check that the cleaning part is working
        # Only 'Do not take this article option'
        for i in range(2):
            article_saver.saved_articles[(ARTICLE_ID, i)] = SAVING_OPTIONS['nothing']

        all_articles = article_saver.saved_articles.keys()
        article_saver.retrieve_text()
        for i in range(2):
            assert (ARTICLE_ID, i) in all_articles
            assert article_saver.df_chosen_texts.loc[(article_saver.df_chosen_texts.article_id == ARTICLE_ID) &
                                                     (article_saver.df_chosen_texts.paragraph_id == i)].empty

        # 'Do not take this article option' and 'Extract the paragraph' options
        ARTICLE_ID = '4vo7n6nh'
        for i in range(2, 4):
            article_saver.saved_articles[(ARTICLE_ID, i)] = SAVING_OPTIONS['paragraph']
        all_articles = article_saver.saved_articles.keys()
        article_saver.retrieve_text()
        for i in range(2, 4):
            assert (ARTICLE_ID, i) in all_articles
            assert not article_saver.df_chosen_texts.loc[(article_saver.df_chosen_texts.article_id == ARTICLE_ID) &
                                                         (article_saver.df_chosen_texts.paragraph_id == i)].empty

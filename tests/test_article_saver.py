import pandas as pd

from bbsearch.article_saver import ArticleSaver
from bbsearch.widget import SAVING_OPTIONS


class TestArticleSaver:

    def test_article_saver(self, fake_db_cursor):
        """Test that article_saver is good. """
        article_saver = ArticleSaver(database=fake_db_cursor)

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
            article_saver.saved_articles[article_id,
                                         all_articles_paragraphs_id[article_id][0]] = 'Extract the paragraph'

        article_saver.saved_articles[article_id,
                                     all_articles_paragraphs_id[article_id][0]] = 'Extract the entire article'

        # Check that the retrieving of the different text is working
        article_saver.retrieve_text()
        assert isinstance(article_saver.articles_text, dict)
        for article_infos, text in article_saver.articles_text.items():
            assert isinstance(text, str)
        assert len(article_saver.articles_text) == len(all_articles_paragraphs_id)

        # Check summary table
        summary_table = article_saver.summary_table()
        assert isinstance(summary_table, pd.DataFrame)

        # Check that the status feedback is working fine
        for article_id, paragraph_id_list in all_articles_paragraphs_id.items():
            assert(article_saver.status_on_article_retrieve((article_id, paragraph_id_list[0])), str)
        assert(article_saver.status_on_article_retrieve((article_id, '1000')), str)

        # Check that the cleaning part is working
        # Only 'Do not take this article option'
        for i in range(2):
            article_saver.saved_articles[('new_id', i)] = SAVING_OPTIONS['nothing']

        all_articles = article_saver.saved_articles.keys()
        cleaned_articles = article_saver.clean_saved_articles().keys()
        for i in range(2):
            assert ('new_id', i) in all_articles
            assert ('new_id', i) not in cleaned_articles

        # 'Do not take this article option' and 'Extract the paragraph' options
        for i in range(2, 4):
            article_saver.saved_articles[('new_id', i)] = SAVING_OPTIONS['paragraph']

        all_articles = article_saver.saved_articles.keys()
        cleaned_articles = article_saver.clean_saved_articles().keys()
        for i in range(4):
            assert ('new_id', i) in all_articles
            if i >= 2:
                assert ('new_id', i) in cleaned_articles
            else:
                assert ('new_id', i) not in cleaned_articles

        # All the options
        article_saver.saved_articles[('new_id', 4)] = SAVING_OPTIONS['article']

        cleaned_articles = article_saver.clean_saved_articles().keys()
        assert ('new_id', None) in cleaned_articles

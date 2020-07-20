import pandas as pd

from bbsearch.widgets import ArticleSaver


class TestArticleSaver:

    def test_article_saver(self, fake_db_cursor, fake_sqlalchemy_engine):
        """Test that article_saver is good. """

        article_saver = ArticleSaver(connection=fake_sqlalchemy_engine)

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
            paragraph_id = all_articles_paragraphs_id[article_id][0]
            # article_saver.saved_articles[article_id, paragraph_id] = SAVING_OPTIONS['paragraph']
            article_saver.add_paragraph(article_id, paragraph_id)

        # For the last article extract all its paragraphs
        # article_saver.saved_articles[article_id, paragraph_id] = SAVING_OPTIONS['article']
        article_saver.add_article(article_id)
        n_paragraphs_full_article = len(all_paragraphs_id)

        # Check that the retrieving of the different text is working
        article_saver.retrieve_text()
        assert isinstance(article_saver.df_chosen_texts, pd.DataFrame)
        assert article_saver.df_chosen_texts.columns.to_list() == ['article_id', 'section_name', 'paragraph_id', 'text']
        assert len(article_saver.df_chosen_texts) == len(all_articles_paragraphs_id) + n_paragraphs_full_article - 1

        # Check summary table
        summary_table = article_saver.summary_table()
        assert isinstance(summary_table, pd.DataFrame)

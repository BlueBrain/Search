import pandas as pd

from bbsearch.widgets import ArticleSaver


class TestArticleSaver:

    def test_adding_removing(self):
        article_saver = ArticleSaver(connection=None)

        full_articles = ["article_1", "article_2", "article_3"]
        just_paragraphs = [
            ("article_1", 0),
            ("article_3", 2),
            ("article_3", 5)]

        # Adding items
        for article_id in full_articles:
            article_saver.add_article(article_id)
        for article_id, paragraph_id in just_paragraphs:
            article_saver.add_paragraph(article_id, paragraph_id)

        # Checking if items were saved
        for article_id in full_articles:
            assert article_saver.has_article(article_id)
        for article_id, paragraph_id in just_paragraphs:
            assert article_saver.has_paragraph(article_id, paragraph_id)

        # Removing items
        article_to_remove = full_articles[0]
        paragraph_to_remove = just_paragraphs[2]

        article_saver.remove_article(article_to_remove)
        assert not article_saver.has_article(article_to_remove)
        article_saver.remove_paragraph(*paragraph_to_remove)
        assert not article_saver.has_paragraph(*paragraph_to_remove)
        article_saver.remove_paragraph("fake_article", 12345)  # doesn't exist

        # Removing all items
        article_saver.remove_all()
        for article_id in full_articles:
            assert not article_saver.has_article(article_id)
        for article_id, paragraph_id in just_paragraphs:
            assert not article_saver.has_paragraph(article_id, paragraph_id)

    def test_summaries(self, fake_db_cursor, fake_sqlalchemy_engine, tmpdir):
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
            article_saver.add_paragraph(article_id, paragraph_id)

        # For the last article extract all its paragraphs
        article_saver.add_article(article_id)
        n_paragraphs_full_article = len(all_paragraphs_id)

        # Check that the retrieving of the different text is working
        df_chosen_texts = article_saver.get_chosen_texts()
        assert isinstance(df_chosen_texts, pd.DataFrame)
        assert df_chosen_texts.columns.to_list() == ['article_id', 'section_name', 'paragraph_id', 'text']
        assert len(df_chosen_texts) == len(all_articles_paragraphs_id) + n_paragraphs_full_article - 1

        # Cached chosen texts
        df_chosen_texts_cached = article_saver.get_chosen_texts()
        assert len(df_chosen_texts) == len(df_chosen_texts_cached)

        # Check summary table
        summary_table = article_saver.summary_table()
        assert isinstance(summary_table, pd.DataFrame)

        # PDF report - probably can't do this on Travis...
        # assert len(tmpdir.listdir()) == 0
        # article_saver.pdf_report(tmpdir)
        # assert len(tmpdir.listdir()) == 1

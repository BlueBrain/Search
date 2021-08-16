"""Tests covering ArticleSaver."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd

from bluesearch.widgets import ArticleSaver


class TestArticleSaver:
    def test_adding_removing(self):
        article_saver = ArticleSaver(connection=None)

        full_articles = np.array([101, 102, 103])
        just_paragraphs = np.array([(101, 0), (103, 2), (103, 5)])

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

        # Test type of IDs in article saver state
        for article_id, paragraph_id in article_saver.state:
            assert type(article_id) == int
            assert type(paragraph_id) == int

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

    def test_summaries(self, fake_sqlalchemy_engine, tmpdir):
        """Test that article_saver is good."""

        article_saver = ArticleSaver(connection=fake_sqlalchemy_engine)

        # Check the possible article_id, paragraphs_id of the fake database
        # Create a fake article_saver.saved_articles dictionary
        # (Which should be the output of the widget)
        sql_query = "SELECT article_id FROM articles"
        article_ids = pd.read_sql(sql_query, fake_sqlalchemy_engine)[
            "article_id"
        ].to_list()
        all_articles_paragraphs_id = {}
        for article_id in set(article_ids):
            sql_query = f"""
            SELECT paragraph_pos_in_article
            FROM sentences
            WHERE article_id = {article_id}
            """
            all_paragraph_pos_in_article = pd.read_sql(
                sql_query, fake_sqlalchemy_engine
            )["paragraph_pos_in_article"].to_list()
            all_articles_paragraphs_id[article_id] = list(
                set(all_paragraph_pos_in_article)
            )
            # For all articles extract only the first of their paragraphs
            paragraph_pos_in_article = all_articles_paragraphs_id[article_id][0]
            article_saver.add_paragraph(article_id, paragraph_pos_in_article)

        # For the last article extract all its paragraphs
        article_saver.add_article(article_id)
        n_paragraphs_full_article = len(set(all_paragraph_pos_in_article))

        # Check that the retrieving of the different text is working
        df_chosen_texts = article_saver.get_chosen_texts()
        assert isinstance(df_chosen_texts, pd.DataFrame)
        assert df_chosen_texts.columns.to_list() == [
            "article_id",
            "section_name",
            "paragraph_pos_in_article",
            "text",
        ]
        assert (
            len(df_chosen_texts)
            == len(all_articles_paragraphs_id) + n_paragraphs_full_article - 1
        )

        # Cached chosen texts
        df_chosen_texts_cached = article_saver.get_chosen_texts()
        assert len(df_chosen_texts) == len(df_chosen_texts_cached)

        # Check summary table
        summary_table = article_saver.summary_table()
        assert isinstance(summary_table, pd.DataFrame)

        # Make report
        assert len(tmpdir.listdir()) == 0
        article_saver.make_report(tmpdir)
        assert len(tmpdir.listdir()) == 1

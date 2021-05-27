"""Module for the article_saver."""

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

import datetime
import pathlib
import textwrap

import pandas as pd

from .._css import style
from ..sql import (
    retrieve_article_metadata_from_article_id,
    retrieve_articles,
    retrieve_paragraph,
)


class ArticleSaver:
    """Keeps track of selected articles.

    This class can be used to save a number of articles and paragraphs
    for a later use. A typical use case is to keep track of the items
    selected in the search widget, and to retrieve them later in the
    mining widget.

    Furthermore this class allows to print a summary of all selected
    items using the `summary_table` method, to resolve all items into
    paragraphs with the corresponding section name and to summarize
    them in a pandas data frame using the method `get_chosen_texts`,
    and to export a PDF report of all saved items using the method
    `report`.

    Parameters
    ----------
    connection : sqlalchemy.engine.Engine
        An SQL database connectable compatible with `pandas.read_sql`.
        The database is supposed to have paragraphs and articles tables.

    Attributes
    ----------
    connection : sqlalchemy.engine.Engine
        An SQL database connectable compatible with `pandas.read_sql`.
        The database is supposed to have paragraphs and articles tables.
    state : set
        The state that keeps track of saved items. It is a set of tuples
        of the form `(article_id, paragraph_id)` each representing one
        saved item. The items with `paragraph_id = -1` indicate that the
        whole article should be saved.
    state_hash : int or None
        A hash uniquely identifying a certain state. This is used to
        cache `df_chosen_texts` and avoid recomputing it if the state
        has not changed.
    df_chosen_texts : pd.DataFrame
        The rows represent different paragraphs and the columns are
        'article_id', 'section_name', 'paragraph_id', 'text'.
    """

    def __init__(self, connection):
        self.connection = connection
        self.state = set()
        self.state_hash = None
        self.df_chosen_texts = pd.DataFrame(
            columns=["article_id", "section_name", "paragraph_pos_in_article", "text"]
        )

    def add_article(self, article_id):
        """Save an article.

        Parameters
        ----------
        article_id : int
            The article ID.
        """
        self.add_paragraph(article_id, -1)

    def add_paragraph(self, article_id, paragraph_pos_in_article):
        """Save a paragraph.

        Parameters
        ----------
        article_id : int
            The article ID.
        paragraph_pos_in_article : int
            The paragraph ID.
        """
        self.state.add((int(article_id), int(paragraph_pos_in_article)))

    def has_article(self, article_id):
        """Check if an article has been saved.

        Parameters
        ----------
        article_id : int
            The article ID.

        Returns
        -------
        result : bool
            Whether or not the given article has been saved.
        """
        return self.has_paragraph(article_id, -1)

    def has_paragraph(self, article_id, paragraph_pos_in_article):
        """Check if a paragraph has been saved.

        Parameters
        ----------
        article_id : int
            The article ID.
        paragraph_pos_in_article : int
            The paragraph ID.

        Returns
        -------
        result : bool
            Whether or not the given paragraph has been saved.
        """
        return (int(article_id), int(paragraph_pos_in_article)) in self.state

    def remove_article(self, article_id):
        """Remove an article from saved.

        Parameters
        ----------
        article_id : int
            The article ID.
        """
        self.remove_paragraph(article_id, -1)

    def remove_paragraph(self, article_id, paragraph_pos_in_article):
        """Remove a paragraph from saved.

        Parameters
        ----------
        article_id : int
            The article ID.
        paragraph_pos_in_article : int
            The paragraph ID.
        """
        if (article_id, paragraph_pos_in_article) in self.state:
            self.state.remove((article_id, paragraph_pos_in_article))

    def remove_all(self):
        """Remove all saved items."""
        self.state.clear()

    def _get_clean_state(self):
        """Get a clean state of the article saver.

        Returns
        -------
        full_articles : set of int
            Set of the article ids chosen by the user.
        just_paragraphs : set of tuple
            Set of tuple (article_id, paragraph_pos_in_article) chosen by the user.
        """
        full_articles = {
            article_id
            for article_id, paragraph_pos_in_article in self.state
            if paragraph_pos_in_article == -1
        }
        just_paragraphs = {
            (article_id, paragraph_pos_in_article)
            for article_id, paragraph_pos_in_article in self.state
            if paragraph_pos_in_article != -1 and article_id not in full_articles
        }
        return full_articles, just_paragraphs

    def get_saved_items(self):
        """Retrieve the saved items that summarize the choice of the users.

        Returns
        -------
        identifiers : list of tuple
            Tuple (article_id, paragraph_pos_in_article) chosen by the user.
        """
        saved_items = []
        full_articles, just_paragraphs = self._get_clean_state()
        for article_id in full_articles:
            saved_items += [(article_id, -1)]
        saved_items += just_paragraphs
        return saved_items

    def _update_chosen_texts(self):
        """Recompute the chosen texts."""
        # empty all rows
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        full_articles, just_paragraphs = self._get_clean_state()

        articles = retrieve_articles(article_ids=full_articles, engine=self.connection)
        self.df_chosen_texts = self.df_chosen_texts.append(articles)

        for (article_id, paragraph_pos_in_article) in just_paragraphs:
            paragraph = retrieve_paragraph(
                article_id, paragraph_pos_in_article, engine=self.connection
            )
            self.df_chosen_texts = self.df_chosen_texts.append(paragraph)

    def get_chosen_texts(self):
        """Retrieve the currently saved items.

        For all entire articles that are saved the corresponding
        paragraphs are resolved first.

        Returns
        -------
        df_chosen_texts : pandas.DataFrame
        """
        state_hash = hash(tuple(sorted(self.state)))
        if state_hash != self.state_hash:
            self._update_chosen_texts()
            self.state_hash = state_hash

        return self.df_chosen_texts.copy()

    def _fetch_article_info(self, article_id):
        article = retrieve_article_metadata_from_article_id(
            article_id=article_id, engine=self.connection
        )
        article_authors, article_title, ref = article.iloc[0][
            ["authors", "title", "url"]
        ]

        return ref, article_title, article_authors

    def make_report(self, output_dir=None):
        """Create the saved articles report.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The directory for writing the report.

        Returns
        -------
        output_file_path : pathlib.Path
            The file to which the report was written.
        """
        css_style = style.get_css_style()
        article_report = f"<style> {css_style} </style>"
        width = 80

        df_chosen_texts = self.get_chosen_texts()

        for article_id, df_article in df_chosen_texts.groupby("article_id"):
            df_article = df_article.sort_values(
                by="paragraph_pos_in_article", ascending=True, axis=0
            )
            if len(df_article["section_name"].unique()) == 1:
                section_name = df_article["section_name"].iloc[0]
            else:
                section_name = (
                    f'{len(df_article["section_name"].unique())} different '
                    f"sections are selected for this article."
                )
            ref, article_title, article_authors = self._fetch_article_info(article_id)
            article_metadata = f"""
            <a href="{ref}">
                <div class="article_title">
                    {article_title}
                </div>
            </a>
            <div class="metadata">
                {article_authors} &#183; {section_name.lower().title()}
            </div>
            """
            article_report += article_metadata

            article_report += "<br/>".join(
                (textwrap.fill(t_, width=width) for t_ in df_article.text)
            )
            article_report += "<br/>" * 2

        if output_dir is None:
            output_dir = pathlib.Path.cwd()
        else:
            output_dir = pathlib.Path(output_dir)
            if not output_dir.exists():
                msg = f"The output directory {output_dir} does not exist."
                raise ValueError(msg)
        output_file_path = (
            output_dir / f"article_saver_report_{datetime.datetime.now()}.html"
        )
        with output_file_path.open("w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(article_report)

        return output_file_path

    def summary_table(self):
        """Create a dataframe table with saved articles.

        Returns
        -------
        table : pd.DataFrame
            DataFrame containing all the paragraphs seen and choice made for it.
        """
        rows = []
        for article_id, paragraph_pos_in_article in self.state:
            if paragraph_pos_in_article == -1:
                option = "Save full article"
            else:
                option = "Save paragraph"
            rows.append(
                {
                    "article_id": article_id,
                    "paragraph_pos_in_article": paragraph_pos_in_article,
                    "option": option,
                }
            )
        table = pd.DataFrame(
            data=rows, columns=["article_id", "paragraph_pos_in_article", "option"]
        )

        return table

"""Module for the article_saver."""
import datetime
import pathlib
import textwrap

import pandas as pd
import pdfkit


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
    connection: SQLAlchemy connectable (engine/connection) or
                database str URI or
                DBAPI2 connection (fallback mode)
        An SQL database connectable compatible with `pandas.read_sql`.
        The database is supposed to have paragraphs and articles tables.

    Attributes
    ----------
    connection: SQLAlchemy connectable (engine/connection) or
                database str URI or
                DBAPI2 connection (fallback mode)
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
        self.df_chosen_texts = pd.DataFrame(columns=[
            'article_id', 'section_name', 'paragraph_id', 'text'])

    def add_article(self, article_id):
        """Save an article.

        Parameters
        ----------
        article_id : str
            The article ID.
        """
        self.add_paragraph(article_id, -1)

    def add_paragraph(self, article_id, paragraph_id):
        """Save a paragraph.

        Parameters
        ----------
        article_id : str
            The article ID.
        paragraph_id : int
            The paragraph ID.
        """
        self.state.add((article_id, paragraph_id))

    def has_article(self, article_id):
        """Check if an article has been saved.

        Parameters
        ----------
        article_id :
            The article ID.

        Returns
        -------
        result : bool
            Whether or not the given article has been saved.
        """
        return self.has_paragraph(article_id, -1)

    def has_paragraph(self, article_id, paragraph_id):
        """Check if a paragraph has been saved.

        Parameters
        ----------
        article_id : str
            The article ID.
        paragraph_id : int
            The paragraph ID.

        Returns
        -------
        result : bool
            Whether or not the given paragraph has been saved.
        """
        return (article_id, paragraph_id) in self.state

    def remove_article(self, article_id):
        """Remove an article from saved.

        Parameters
        ----------
        article_id : str
            The article ID.
        """
        self.remove_paragraph(article_id, -1)

    def remove_paragraph(self, article_id, paragraph_id):
        """Remove a paragraph from saved.

        Parameters
        ----------
        article_id : str
            The article ID.
        paragraph_id : int
            The paragraph ID.
        """
        if (article_id, paragraph_id) in self.state:
            self.state.remove((article_id, paragraph_id))

    def remove_all(self):
        """Remove all saved items."""
        self.state.clear()

    def _update_chosen_texts(self):
        """Recompute the chosen texts."""
        # empty all rows
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        full_articles = set(article_id
                            for article_id, paragraph_id in self.state
                            if paragraph_id == -1)
        just_paragraphs = set(paragraph_id
                              for article_id, paragraph_id in self.state
                              if paragraph_id != -1 and
                              article_id not in full_articles)

        article_ids_full_list = ','.join(f"\"{id_}\"" for id_ in full_articles)
        sql_query = f"""
        SELECT article_id, section_name, paragraph_id, text
        FROM (
                 SELECT *
                 FROM paragraphs
                 WHERE sha IN (
                     SELECT sha
                     FROM article_id_2_sha
                     WHERE article_id IN ({article_ids_full_list})
                 )
             ) p
                 INNER JOIN
             article_id_2_sha a
             ON a.sha = p.sha;
        """
        df_extractions_full = pd.read_sql(sql_query, self.connection)

        paragraph_ids_list = ','.join(f"\"{id_}\"" for id_ in just_paragraphs)
        sql_query = f"""
        SELECT article_id, section_name, paragraph_id, text
        FROM (
                 SELECT *
                 FROM paragraphs
                 WHERE paragraph_id IN ({paragraph_ids_list})
             ) p
                 INNER JOIN
             article_id_2_sha a
             ON p.sha = a.sha;
        """
        df_extractions_pars = pd.read_sql(sql_query, self.connection)

        self.df_chosen_texts = df_extractions_full.append(df_extractions_pars, ignore_index=True)

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
        sql_query = f"""
        SELECT authors, title, url
        FROM articles
        WHERE article_id = "{article_id}"
        """
        article = pd.read_sql(sql_query, self.connection)
        article_authors, article_title, ref = \
            article.iloc[0][['authors', 'title', 'url']]

        return ref, article_title, article_authors

    def pdf_report(self, output_dir):
        """Create the saved articles report.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The directory for writing the report.

        Returns
        -------
        output_file_path : str
            The file to which the report was written.
        """
        article_report = ''
        width = 80

        df_chosen_texts = self.get_chosen_texts()

        color_title = '#1A0DAB'
        color_metadata = '#006621'
        for article_id, df_article in df_chosen_texts.groupby('article_id'):
            df_article = df_article.sort_values(by='paragraph_id', ascending=True, axis=0)
            if len(df_article['section_name'].unique()) == 1:
                section_name = df_article['section_name'].iloc[0]
            else:
                section_name = f'{len(df_article["section_name"].unique())} different ' \
                               f'sections are selected for this article.'
            ref, article_title, article_authors = self._fetch_article_info(article_id)
            article_metadata = f"""
            <a href="{ref}" style="color:{color_title}; font-size:17px">
                {article_title}
            </a>
            <br>
            <p style="color:{color_metadata}; font-size:13px">
                {article_authors} &#183; {section_name.lower().title()}
            </p>
            """
            article_report += article_metadata

            article_report += '<br/>'.join((textwrap.fill(t_, width=width) for t_ in df_article.text))
            article_report += '<br/>' * 2

        if output_dir is None:
            output_dir = pathlib.Path.cwd()
        else:
            output_dir = pathlib.Path(output_dir)
            if not output_dir.exists():
                msg = f"The output directory {output_dir} does not exist."
                raise ValueError(msg)
        filename = f"article_saver_report_{datetime.datetime.now()}.pdf"
        output_file_path = str(output_dir / filename)
        pdfkit.from_string(article_report, output_file_path)

        return output_file_path

    def summary_table(self):
        """Create a dataframe table with saved articles.

        Returns
        -------
        table: pd.DataFrame
            DataFrame containing all the paragraphs seen and choice made for it.
        """
        rows = []
        for article_id, paragraph_id in self.state:
            if paragraph_id == -1:
                option = "Save full article"
            else:
                option = "Save paragraph"
            rows.append({
                'article_id': article_id,
                'paragraph_id': paragraph_id,
                'option': option})
        table = pd.DataFrame(
            data=rows,
            columns=['article_id', 'paragraph_id', 'option'])

        return table

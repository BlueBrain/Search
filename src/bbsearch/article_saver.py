"""Module for the article_saver."""
import datetime
import pdfkit
import textwrap

import pandas as pd

from .sql import get_shas_from_ids
from .widget import SAVING_OPTIONS


class ArticleSaver:
    """Articles saved used to link Search Engine and Entities/Relation Extraction.

    Parameters
    ----------
    database: sqlite3.Connection
        Connection to the database. The database is supposed to have paragraphs and
        articles tables.
    """

    def __init__(self,
                 database):
        self.db = database
        self.saved_articles = dict()
        self.df_chosen_texts = pd.DataFrame(columns=['article_id', 'section_name', 'paragraph_id', 'text'])
        self.articles_metadata = dict()

    def retrieve_text(self):
        """Retrieve text of every article given the option chosen by the user."""
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        df_all_options = pd.DataFrame.from_records(data=[(*k, v) for k, v in self.saved_articles.items()],
                                                   columns=['article_id', 'paragraph_id', 'option'])

        for article_id in df_all_options['article_id'].unique():
            df_article_options = df_all_options.loc[df_all_options['article_id'] == article_id]
            if (SAVING_OPTIONS['article'] in df_article_options['option'].values) or \
                    (SAVING_OPTIONS['paragraph'] in df_article_options['option'].values):
                if SAVING_OPTIONS['article'] in df_article_options['option'].values:
                    shas = get_shas_from_ids([article_id], self.db)
                    if len(shas) == 1:
                        condition = f'WHERE sha = {repr(shas[0])}'
                    elif len(shas) > 1:
                        condition = f'WHERE sha IN {repr(tuple(shas))}'
                    else:
                        raise ValueError(f'No sha could be found for article_id={article_id}')
                else:
                    paragraphs_to_extract = df_article_options.loc[
                        df_article_options['option'] == SAVING_OPTIONS['paragraph']]['paragraph_id'].values
                    if len(paragraphs_to_extract) == 1:
                        condition = f'WHERE paragraph_id = {repr(paragraphs_to_extract[0])}'
                    elif len(paragraphs_to_extract) > 1:
                        condition = f'WHERE paragraph_id IN {repr(tuple(paragraphs_to_extract))}'
                query = f'''SELECT section_name, paragraph_id, text
                                    FROM paragraphs
                                    {condition}
                                    ORDER BY paragraph_id ASC'''
                df_paragraphs_sections = pd.read_sql(query, self.db)
                self.df_chosen_texts = self.df_chosen_texts.append(pd.DataFrame(data={
                    'article_id': [article_id] * len(df_paragraphs_sections),
                    **df_paragraphs_sections
                }), ignore_index=True)

    def report(self):
        """Create the saved articles report.

        Returns
        -------
        path: str
            Path where the report is generated
        """
        article_report = ''
        width = 80

        self.retrieve_text()
        for article_id, df_article in self.df_chosen_texts.groupby('article_id'):
            df_article = df_article.sort_values(by='paragraph_id', ascending=True, axis=0)
            if len(df_article['section_name'].unique()) == 1:
                article_report += self.articles_metadata[article_id]
            else:
                substring = '&#183;'
                article_report += self.articles_metadata[article_id].split(substring)[0] + '&#183;'
                article_report += f'{len(df_article["section_name"].unique())} different ' \
                                  f'sections are selected for this article.'
                article_report += '</p>'
            article_report += '<br/>'.join((textwrap.fill(t_, width=width) for t_ in df_article.text))
            article_report += '<br/>' * 2

        path = f"report_{datetime.datetime.now()}.pdf"
        pdfkit.from_string(article_report, path)
        return path

    def summary_table(self):
        """Create a dataframe table with saved articles.

        Returns
        -------
        table: pd.DataFrame
            DataFrame containing all the paragraphs seen and choice made for it.
        """
        articles = []
        for article_infos, option in self.saved_articles.items():
            articles += [{'article_id': article_infos[0],
                          'paragraph_id': article_infos[1],
                          'option': option
                          }]
        table = pd.DataFrame(data=articles,
                             columns=['article_id', 'paragraph_id', 'option'])
        table.sort_values(by=['article_id'])
        return table

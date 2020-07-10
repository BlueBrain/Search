"""Module for the article_saver."""
import datetime
import pdfkit
import textwrap

import pandas as pd

from .widget import SAVING_OPTIONS


class ArticleSaver:
    """Articles saved used to link Search Engine and Entities/Relation Extraction.

    Parameters
    ----------
    connection: SQLAlchemy connectable (engine/connection) or database str URI
        An SQL database connectable compatible with `pandas.read_sql`.
        The database is supposed to have paragraphs and articles tables.

    Attributes
    ----------
    saved_articles : dict
        The keys are a tuple where the first element is the article_id and the second
        is the paragraph_id. The values are one of the 3 options below:
        - 'Do not take this article'
        - 'Extract the paragraph'
        - 'Extract the entire article'

    df_chosen_texts : pd.DataFrame
        The rows represent different paragraphs and the columns are 'article_id', 'section_name',
        'paragraph_id', 'text'.

    articles_metadata : dict
        The keys are article_ids and the value is a string (HTML formatting) with title,
        authors, etc.

    """

    def __init__(self, connection):
        self.connection = connection
        self.saved_articles = dict()
        self.df_chosen_texts = pd.DataFrame(columns=['article_id', 'section_name', 'paragraph_id', 'text'])
        self.articles_metadata = dict()

    def retrieve_text(self):
        """Retrieve text of every article given the option chosen by the user."""
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        df_all_options = pd.DataFrame.from_records(data=[(*k, v) for k, v in self.saved_articles.items()],
                                                   columns=['article_id', 'paragraph_id', 'option'])

        article_ids_full = df_all_options.loc[df_all_options['option'] == SAVING_OPTIONS['article'], 'article_id']
        article_ids_full_list = ','.join(f"\"{id_}\"" for id_ in article_ids_full)

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

        df_only_paragraph = df_all_options.loc[~df_all_options['article_id'].isin(article_ids_full)]
        df_only_paragraph = df_only_paragraph.loc[df_only_paragraph['option'] == SAVING_OPTIONS['paragraph']]

        paragraph_ids_list = ','.join(f"\"{id_}\"" for id_ in df_only_paragraph['paragraph_id'])

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

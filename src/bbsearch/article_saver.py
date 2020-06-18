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

    def status_on_article_retrieve(self, article_infos):
        """Send status about an article given the article_infos (article_id, paragraph_id).

        Parameters
        ----------
        article_infos: tuple
            Tuple (article_id, paragraph_id) of a given paragraph.

        Returns
        -------
        status: str
            String explaining if the given article has already been seen,
            and if yes which option has been chosen by the user.
        """
        status = 'You have never seen this article'
        if article_infos in self.saved_articles.keys():
            status = f'You have already seen this paragraph and ' \
                     f'you chose the option: {self.saved_articles[article_infos]}.'
            return status
        if article_infos[0] in [k[0] for k in self.saved_articles.keys()]:
            status = 'You have already seen this article through different paragraphs'

        return status

    def retrieve_text(self):
        """Retrieve text of every article given the option chosen by the user."""
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        df_all_options = pd.DataFrame.from_records(data=[(*k, v) for k, v in self.saved_articles.items()],
                                                   columns=['article_id', 'paragraph_id', 'option'])

        article_ids_full = df_all_options.loc[df_all_options['option'] == SAVING_OPTIONS['article'], 'article_id']
        article_ids_full_list = ','.join((f"\"{id_}\"" for id_ in article_ids_full))

        sql_query = f"""
        SELECT a.article_id, p.section_name, p.paragraph_id, p.text
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
        df_extractions_full = pd.read_sql(sql_query, self.db)

        df_only_paragraph = df_all_options.loc[~df_all_options['article_id'].isin(article_ids_full)]
        df_only_paragraph = df_only_paragraph.loc[df_only_paragraph['option'] == SAVING_OPTIONS['paragraph']]

        paragraph_ids_list = ','.join((f"\"{id_}\"" for id_ in df_only_paragraph['paragraph_id']))

        sql_query = f"""
        SELECT a.article_id, p.section_name, p.paragraph_id, p.text
        FROM (
                 SELECT *
                 FROM paragraphs
                 WHERE paragraph_id IN ({paragraph_ids_list})
             ) p
                 INNER JOIN
             article_id_2_sha a
             ON p.sha = a.sha;
        """
        df_extractions_pars = pd.read_sql(sql_query, self.db)

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
            article_report += self.articles_metadata[article_id]
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

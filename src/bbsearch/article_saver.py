"""Module for the article_saver."""
import datetime
import pdfkit
import textwrap

import pandas as pd
import sqlalchemy

from .widget import SAVING_OPTIONS


class ArticleSaver:
    """Articles saved used to link Search Engine and Entities/Relation Extraction.

    Parameters
    ----------
    engine: SQLAlchemy.Engine
        Engine connected to the database. The database is supposed to have paragraphs and
        articles tables.

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

    def __init__(self,
                 engine):
        self.engine = engine
        metadata = sqlalchemy.MetaData()
        self.connection = engine.connect()

        self.sentences = sqlalchemy.Table('sentences', metadata,
                                          autoload=True, autoload_with=self.engine)
        self.paragraphs = sqlalchemy.Table('paragraphs', metadata,
                                           autoload=True, autoload_with=self.engine)
        self.articles = sqlalchemy.Table('articles', metadata,
                                         autoload=True, autoload_with=self.engine)
        self.article_id_2_sha = sqlalchemy.Table('article_id_2_sha', metadata,
                                                 autoload=True, autoload_with=self.engine)

        self.saved_articles = dict()
        self.df_chosen_texts = pd.DataFrame(columns=['article_id', 'section_name', 'paragraph_id', 'text'])
        self.articles_metadata = dict()

    def retrieve_text(self):
        """Retrieve text of every article given the option chosen by the user."""
        self.df_chosen_texts = self.df_chosen_texts[0:0]

        df_all_options = pd.DataFrame.from_records(data=[(*k, v) for k, v in self.saved_articles.items()],
                                                   columns=['article_id', 'paragraph_id', 'option'])

        article_ids_full = df_all_options.loc[df_all_options['option'] == SAVING_OPTIONS['article'], 'article_id']

        shas = sqlalchemy.select(
            [self.article_id_2_sha.c.sha]
        ).where(self.article_id_2_sha.c.article_id.in_(article_ids_full))

        p = sqlalchemy.select(
            [self.paragraphs.c.paragraph_id,
             self.paragraphs.c.sha.label('p_sha'),
             self.paragraphs.c.section_name,
             self.paragraphs.c.text]
        ).where(self.paragraphs.c.sha.in_(shas))

        query = sqlalchemy.select(
            [self.article_id_2_sha.c.article_id,
             p.c.section_name,
             p.c.paragraph_id,
             p.c.text]
        )
        query = query.select_from(
            p.join(self.article_id_2_sha,
                   p.c.p_sha == self.article_id_2_sha.c.sha
                   )
        )
        df_extractions_full = pd.read_sql(query, self.connection)

        df_only_paragraph = df_all_options.loc[~df_all_options['article_id'].isin(article_ids_full)]
        df_only_paragraph = df_only_paragraph.loc[df_only_paragraph['option'] == SAVING_OPTIONS['paragraph']]

        p = sqlalchemy.select(
            [self.paragraphs.c.paragraph_id,
             self.paragraphs.c.sha.label('p_sha'),
             self.paragraphs.c.section_name,
             self.paragraphs.c.text]
        ).where(self.paragraphs.c.paragraph_id.in_(df_only_paragraph['paragraph_id'].tolist()))

        query = sqlalchemy.select(
            [self.article_id_2_sha.c.article_id,
             p.c.section_name,
             p.c.paragraph_id,
             p.c.text])

        query = query.select_from(
            p.join(self.article_id_2_sha,
                   p.c.p_sha == self.article_id_2_sha.c.sha
                   ))
        df_extractions_pars = pd.read_sql(query, self.connection)

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

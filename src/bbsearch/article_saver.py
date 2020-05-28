"""Module for the article_saver."""
from collections import defaultdict
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
    database: sqlite3.Cursor
        Cursor to the database. The database is supposed to have paragraphs and
        articles tables.
    """

    def __init__(self,
                 database):

        self.db = database

        self.saved_articles = dict()

        self.articles_text = dict()
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
            status = f'You have already seen this article through different paragraphs'

        return status

    def clean_saved_articles(self):
        """Clean the dictionary saved_articles.

        This function is cleaning the selection of articles with the most generous assumption:
        'Extract the entire article' > 'Extract the paragraph' > 'Do not take this article'
        - If, for a given article_id, 'Extract the entire article' has been chosen,
        the entire article is kept. (even if the two others options have also been chosen)
        - If, for a given article_id, 'Extract the entire article' has never been chosen
        but 'Extract paragraph' has been at least once, all the paragraphs selected are kept.
        - It means that 'Do not take this article' is only taken into account, if it is the
        choice each time the user saw this particular article (through the article_id)

        Returns
        -------
        cleaned_saved_articles: dict
            Clean dictionary of all the articles/paragraphs to keep.
        """
        cleaned_saved_articles = dict()
        articles_id_dict = defaultdict(set)

        for article_infos, option in self.saved_articles.items():
            articles_id_dict[article_infos[0]].add(option)

        for article_id, option_set in articles_id_dict.items():
            if SAVING_OPTIONS['article'] in option_set:
                cleaned_saved_articles[(article_id, None)] = SAVING_OPTIONS['article']
            elif SAVING_OPTIONS['paragraph'] in option_set:
                paragraphs_id = [article_infos[1] for article_infos, option in self.saved_articles.items()
                                 if article_infos[0] == article_id and
                                 option == SAVING_OPTIONS['paragraph']]
                for paragraph_id in paragraphs_id:
                    cleaned_saved_articles[(article_id, paragraph_id)] = SAVING_OPTIONS['paragraph']
        return cleaned_saved_articles

    def extract_entire_article(self, article_id):
        """Extract the entire article text of a given article_id.

        Parameters
        ----------
        article_id: str
            Article_id for the article text to retrieve.

        Returns
        -------
        entire_article: str
            Text of the specified article_id
        """
        entire_article = ''
        all_paragraphs = dict()
        shas = get_shas_from_ids([article_id, ], self.db)
        for sha in shas:
            query_execution = self.db.execute(
                """SELECT paragraph_id, text
                FROM paragraphs WHERE sha = ? ORDER BY paragraph_id ASC""", [sha])
            results = query_execution.fetchone()
            while results is not None:
                paragraph_id, paragraph = results
                all_paragraphs[paragraph_id] = paragraph
                results = query_execution.fetchone()

        for _, text in sorted(all_paragraphs.items()):
            entire_article += text + '\n\n'

        return entire_article

    def extract_paragraph(self, paragraph_id):
        """Extract paragraphs for a given paragraph_id.

        Parameters
        ----------
        paragraph_id: int or str
            Paragraph_id for the paragraph to retrieve.

        Returns
        -------
        paragraph: str
            Text of the paragraph specified.
        """
        (paragraph, ) = self.db.execute(
            """SELECT text FROM paragraphs WHERE paragraph_id = ?""", [paragraph_id]).fetchall()[0]
        return paragraph

    def retrieve_text(self):
        """Retrieve text of every article given the option chosen by the user."""
        self.articles_text.clear()

        clean_saved_articles = self.clean_saved_articles()

        for article_infos, option in clean_saved_articles.items():
            if SAVING_OPTIONS['paragraph'] == option:
                paragraph = self.extract_paragraph(article_infos[1])
                self.articles_text[article_infos] = paragraph
            elif SAVING_OPTIONS['article'] == option:
                article = self.extract_entire_article(article_infos[0])
                self.articles_text[article_infos] = article

    def report(self):
        """Create the saved articles report.

        Returns
        -------
        path: str
            Path where the report is generated
        """
        print("Saving articles results to a pdf file.")
        article_report = ''
        width = 80

        self.retrieve_text()
        for article_infos, text in self.articles_text.items():
            article_report += self.articles_metadata[article_infos[0]]
            article_report += textwrap.fill(text, width=width)
            article_report += '<br/>' + '<br/>'

        path = f"report_{datetime.datetime.now()}.pdf"
        pdfkit.from_string(article_report, path)
        print('Report Generated')
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
                          'choice': option,
                          'paragraph': self.extract_paragraph(article_infos[1])}]
        table = pd.DataFrame(data=articles,
                             columns=['article_id', 'choice', 'paragraph'])
        table.sort_values(by=['article_id'])
        return table

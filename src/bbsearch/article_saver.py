"""Module for the article_saver."""
from collections import defaultdict

from .sql import get_shas_from_ids


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
        self.options = {'Do not take this article': 'nothing',
                        'Extract the paragraph': 'paragraph',
                        'Extract the entire article': 'article'}

        self.articles_text = dict()
        self.articles_metadata = dict()

    def status_on_article_retrieve(self,
                                   article_infos):
        """Send status about an article given the article_infos (article_id, paragraph_id).

        Parameters
        ----------
        article_infos: tuple
            Tuple (article_id, paragraph_id) of a given paragraph.

        Returns
        -------
        status: str
            String in HTML format explaining if the given article has already been seen,
            and if yes which option has been chosen by the user.
        """
        color_text = '#bdbdbd'
        color_highlight = '#a8a8a8'

        status = f"""<p style="font-size:13px; color:{color_text}">
        You have never seen this article </p>"""
        if article_infos in self.saved_articles.keys():
            status = f"""<p style="font-size:13px; color:{color_text}">
            You have already seen this paragraph and you chose the option
            <b style="color:{color_highlight}"> {self.saved_articles[article_infos][0]} </b> </p>"""
            return status

        if article_infos[0] in [k[0] for k in self.saved_articles.keys()]:
            status = f"""<p style="font-size:13px; color:{color_text}">
            You have already seen this article through different paragraphs </p>"""

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
            if 'Extract the entire article' in option_set:
                cleaned_saved_articles[(article_id, None)] = 'Extract the entire article'
                continue
            elif 'Extract the paragraph' in option_set:
                paragraphs_id = [article_infos[1] for article_infos, option in self.saved_articles.items()
                                 if article_infos[0] == article_id and
                                 option == 'Extract the paragraph']
                for paragraph_id in paragraphs_id:
                    cleaned_saved_articles[(article_id, paragraph_id)] = 'Extract the paragraph'
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
            query_end = False
            while not query_end:
                results = query_execution.fetchone()
                if results is not None:
                    paragraph_id, paragraph = results
                    all_paragraphs[paragraph_id] = paragraph
                else:
                    query_end = True

        for _, text in sorted(all_paragraphs.items()):
            entire_article += text

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
        print(clean_saved_articles)

        for article_infos, option in clean_saved_articles.items():
            if self.options[option] == 'paragraph':
                paragraph = self.extract_paragraph(article_infos[1])
                self.articles_text[article_infos] = paragraph
            elif self.options[option] == 'article':
                article = self.extract_entire_article(article_infos[0])
                self.articles_text[article_infos] = article

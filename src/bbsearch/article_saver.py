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

        self.articles_text = list()
        self.report = ''

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
        """Clean the dictionary saved_articles."""
        pass

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
        """Extracts paragraphs for a given paragraph_id

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
        """Retrieve text of every saved article."""
        self.clean_saved_articles()
        self.articles_text = list()
        for article_infos, option in self.saved_articles.items():
            if self.options[option[0]] == 'paragraph':
                paragraph = self.extract_paragraph(article_infos[1])
                self.articles_text.append((option[1], paragraph))
            elif self.options[option[0]] == 'article':
                article = self.extract_entire_article(article_infos[0])
                self.articles_text.append((option[1], article))

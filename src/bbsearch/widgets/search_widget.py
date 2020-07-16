"""The Search widget."""
import collections
import datetime
import functools
import logging
import math
import pdfkit
import textwrap

from IPython.display import display, HTML
import ipywidgets as widgets
import pandas as pd

from ..sql import find_paragraph

logger = logging.getLogger(__name__)

SAVING_OPTIONS = collections.OrderedDict([
    ('nothing',  'Do not take this article'),
    ('paragraph', 'Extract the paragraph'),
    ('article', 'Extract the entire article')
])


class SearchWidget(widgets.VBox):
    """Widget for search engine.

    Parameters
    ----------
    searcher : bbsearch.search.LocalSearcher or bbsearch.remote_searcher.RemoteSearcher
        The search engine.

    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the SQL database

    article_saver: ArticleSaver, optional
        If specified, this article saver will keep all the article_id
        of interest for the user during the different queries.

    results_per_page : int, optional
        The number of results to display per results page.
    """

    def __init__(self,
                 searcher,
                 connection,
                 article_saver=None,
                 results_per_page=10):
        super().__init__()

        self.searcher = searcher
        self.connection = connection
        self.article_saver = article_saver
        self.results_per_page = max(1, results_per_page)
        self.n_pages = 1
        self.current_page = -1

        self.report = ''

        self.radio_buttons = []
        self.current_sentence_ids = []
        self.saving_options = list(SAVING_OPTIONS.values())

        self.widgets = dict()
        self._init_widgets()
        self._adjust_widgets()
        self._init_ui()

    def _init_widgets(self):
        """Initialize widget dictionary."""
        # Select model to compute Sentence Embeddings
        self.widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['USE', 'SBERT', 'BSV', 'SBioBERT'],
            description='Model for Sentence Embedding',
            tooltips=['Universal Sentence Encoder', 'Sentence BERT', 'BioSentVec',
                      'Sentence BioBERT'], )

        # Select n. of top results to return
        self.widgets['top_results'] = widgets.widgets.IntSlider(
            value=10,
            min=0,
            max=100,
            description='Top N results')

        # Choose whether to print whole paragraph containing sentence highlighted, or just the
        # sentence
        self.widgets['print_paragraph'] = widgets.Checkbox(
            value=True,
            description='Show whole paragraph')

        # Enter Query
        self.widgets['query_text'] = widgets.Textarea(
            value='Glucose is a risk factor for COVID-19',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Query')

        self.widgets['has_journal'] = widgets.Checkbox(
            description="Require Journal",
            value=False)

        self.widgets['date_range'] = widgets.IntRangeSlider(
            description="Date Range:",
            continuous_update=False,
            min=1900,
            max=2020,
            value=(1900, 2020),
            layout=widgets.Layout(width='80ch'))

        # Enter Deprioritization Query
        self.widgets['deprioritize_text'] = widgets.Textarea(
            value='',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Deprioritize')

        # Select Deprioritization Strength
        self.widgets['deprioritize_strength'] = widgets.ToggleButtons(
            options=['None', 'Weak', 'Mild', 'Strong', 'Stronger'],
            disabled=False,
            button_style='info',
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength', )

        # Enter Substrings Exclusions
        self.widgets['exclusion_text'] = widgets.Textarea(
            layout=widgets.Layout(width='90%', height='80px'),
            value='',
            style={'description_width': 'initial'},
            description='Substring Exclusion (newline separated): ')

        self.widgets['default_value_article_saver'] = widgets.ToggleButtons(
            options=self.saving_options,
            disabled=False,
            style={'description_width': 'initial', 'button_width': '200px'},
            description='Default saving: ', )
        self.widgets['default_saving_paragraph'] = widgets.Checkbox(
            description="Save paragraph",
            value=False,
            indent=False)
        self.widgets['default_saving_article'] = widgets.Checkbox(
            description="Save article",
            value=False,
            indent=False)

        def default_value_article_saver_change(change):
            if self.radio_buttons:
                for article_infos, button in self.radio_buttons:
                    button.value = change['new']
                return change['new']

        self.widgets['default_value_article_saver'].observe(default_value_article_saver_change,
                                                            names='value')

        # Click to run Information Retrieval!
        self.widgets['investigate_button'] = widgets.Button(description='Investigate!',
                                                            layout=widgets.Layout(width='50%'))

        # Click to run Generate Report!
        self.widgets['report_button'] = widgets.Button(description='Generate Report of Search Results',
                                                       layout=widgets.Layout(width='50%'))

        self.widgets['articles_button'] = widgets.Button(description='Generate Report of Selected Articles',
                                                         layout=widgets.Layout(width='50%'))
        # Output Area
        self.widgets['out'] = widgets.Output(layout={'border': '1px solid black'})

        # Page buttons
        self.widgets['page_back'] = widgets.Button(
            description="←", layout={'width': 'auto'})
        self.widgets['page_label'] = widgets.Label(value="Page - of -")
        self.widgets['page_forward'] = widgets.Button(
            description="→", layout={'width': 'auto'})
        self.widgets['page_back'].on_click(
            lambda b: self.set_page(self.current_page - 1))
        self.widgets['page_forward'].on_click(
            lambda b: self.set_page(self.current_page + 1))

        # Callbacks
        self.widgets['investigate_button'].on_click(self.investigate_on_click)
        self.widgets['report_button'].on_click(self.report_on_click)
        self.widgets['articles_button'].on_click(self.article_report_on_click)

    def _adjust_widgets(self):
        """Hide from the user not used functionalities in the widgets."""
        self.widgets['exclusion_text'].layout.display = 'none'
        # Remove some models (USE and SBERT)
        self.widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['BSV', 'SBioBERT'],
            description='Model for Sentence Embedding',
            tooltips=['BioSentVec', 'Sentence BioBERT'], )
        # Remove some deprioritization strength
        self.widgets['deprioritize_strength'] = widgets.ToggleButtons(
            options=['None', 'Mild', 'Stronger'],
            disabled=False,
            button_style='info',
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength')

    def _init_ui(self):
        page_selection = widgets.HBox(children=[
                self.widgets['page_back'],
                self.widgets['page_label'],
                self.widgets['page_forward']
        ])
        self.children = [
            self.widgets['sent_embedder'],
            self.widgets['top_results'],
            self.widgets['print_paragraph'],
            self.widgets['query_text'],
            self.widgets['has_journal'],
            self.widgets['date_range'],
            self.widgets['deprioritize_text'],
            self.widgets['deprioritize_strength'],
            self.widgets['exclusion_text'],
            self.widgets['default_value_article_saver'],
            self.widgets['default_saving_paragraph'],
            self.widgets['default_saving_article'],
            self.widgets['investigate_button'],
            page_selection,
            self.widgets['out'],
            page_selection,
            self.widgets['report_button'],
            self.widgets['articles_button'],
        ]

        with self.widgets['out']:
            init_text = """
              ____  ____   _____ 
             |  _ \|  _ \ / ____|
             | |_) | |_) | (___  
             |  _ <|  _ < \___ \ 
             | |_) | |_) |____) |
             |____/|____/|_____/ 
                                               
            Click \"Investiage\" to display some results.
            """
            print(textwrap.dedent(init_text))

    @staticmethod
    def highlight_in_paragraph(paragraph, sentence):
        """Highlight a given sentence in the paragraph.

        Parameters
        ----------
        paragraph : str
            The paragraph in which to highlight the sentence.
        sentence: str
            The sentence to highlight.

        Returns
        -------
        formatted_paragraph : str
            The paragraph containing `sentence` with the sentence highlighted
            in color
        """
        color_text = '#222222'
        color_highlight = '#000000'

        start = paragraph.index(sentence)
        end = start + len(sentence)
        highlighted_paragraph = f"""
            <p style="font-size:13px; color:{color_text}">
            {paragraph[:start]}
            <b style="color:{color_highlight}"> {paragraph[start:end]} </b>
            {paragraph[end:]}
            </p>
            """

        return highlighted_paragraph

    def print_single_result(self, sentence_id, print_whole_paragraph):
        """Retrieve metadata and complete the report with HTML string given sentence_id.

        Parameters
        ----------
        sentence_id: int
            Sentence ID of the article needed to retrieve
        print_whole_paragraph: bool
            If true, the whole paragraph will be displayed in the results of the widget.

        Returns
        -------
        article_metadata: str
            Formatted string containing the metadata of the article.
        formatted_output: str
            Formatted output of the sentence.
        article_id : str
            The article ID.
        paragraph_id : int
            The paragraph ID.
        """
        sql_query = f"""
        SELECT sha, section_name, text, paragraph_id
        FROM sentences
        WHERE sentence_id = "{sentence_id}"
        """
        sentence = pd.read_sql(sql_query, self.connection)
        article_sha, section_name, text, paragraph_id = \
            sentence.iloc[0][['sha', 'section_name', 'text', 'paragraph_id']]

        sql_query = f"""
        SELECT article_id
        FROM article_id_2_sha
        WHERE sha = "{article_sha}"
        """
        article_id = pd.read_sql(sql_query, self.connection).iloc[0]['article_id']

        sql_query = f"""
        SELECT authors, title, url
        FROM articles
        WHERE article_id = "{article_id}"
        """
        article = pd.read_sql(sql_query, self.connection)
        article_auth, article_title, ref = \
            article.iloc[0][['authors', 'title', 'url']]

        try:
            article_auth = article_auth.split(';')[0] + ' et al.'
        except AttributeError:
            article_auth = ''

        ref = ref or ""
        section_name = section_name or ""

        width = 80
        if print_whole_paragraph:
            try:
                paragraph = find_paragraph(sentence_id, self.connection)
                formatted_output = self.highlight_in_paragraph(
                    paragraph, text)
            except Exception as err:
                formatted_output = f"""
                There was a problem retrieving the paragraph.
                The original sentence is: {text}
                The error was: {str(err)}
                """
        else:
            formatted_output = textwrap.fill(text, width=width)

        color_title = '#1A0DAB'
        color_metadata = '#006621'
        article_metadata = f"""
            <a href="{ref}" style="color:{color_title}; font-size:17px">
                {article_title}
            </a>
            <br>
            <p style="color:{color_metadata}; font-size:13px">
                {article_auth} &#183; {section_name.lower().title()}
            </p>
            """
        article_metadata = textwrap.dedent(article_metadata)

        return article_metadata, formatted_output, article_id, paragraph_id

    def investigate_on_click(self, change_dict):
        """Investigate button callback."""

        # Get user selection
        which_model = self.widgets['sent_embedder'].value
        k = self.widgets['top_results'].value
        query_text = self.widgets['query_text'].value
        deprioritize_text = self.widgets['deprioritize_text'].value
        deprioritize_strength = self.widgets['deprioritize_strength'].value
        exclusion_text = self.widgets['exclusion_text'].value \
            if 'exclusion_text' in self.widgets.keys() else ''
        has_journal = self.widgets['has_journal'].value
        date_range = self.widgets['date_range'].value

        # Clear output and show waiting message
        self.widgets['out'].clear_output()
        with self.widgets['out']:
            print(f'Processing query \"{query_text}\"...')
            print('Sending query to server...')

            # Perform search query
            self.current_sentence_ids, *_ = self.searcher.query(
                which_model=which_model,
                k=k,
                query_text=query_text,
                has_journal=has_journal,
                date_range=date_range,
                deprioritize_strength=deprioritize_strength,
                deprioritize_text=deprioritize_text,
                exclusion_text=exclusion_text)

            print('Updating the results display...')
            self.n_pages = math.ceil(
                len(self.current_sentence_ids) / self.results_per_page)

        # Update the results display
        self.set_page(0, force=True)

    def set_page(self, new_page, force=False):
        new_page = max(0, min(new_page, self.n_pages - 1))
        if self.current_page != new_page or force:
            self.current_page = new_page
            page_label = f'Page {self.current_page + 1} of {self.n_pages}'
            self.widgets['page_label'].value = page_label
            self._update_page_display()

    def _update_page_display(self):
        with self.widgets['out']:
            print_whole_paragraph = self.widgets['print_paragraph'].value
            self.radio_buttons = list()

            self.widgets['out'].clear_output()
            start = self.current_page * self.results_per_page
            end = start + self.results_per_page
            for sentence_id in self.current_sentence_ids[start:end]:
                if self.article_saver:
                    article_metadata, formatted_output, article_id, paragraph_id = \
                        self.print_single_result(int(sentence_id), print_whole_paragraph)

                    # radio_button = self.create_radio_buttons((article_id, paragraph_id), article_metadata)
                    if self.widgets['default_saving_paragraph'].value is True:
                        self.article_saver.add_paragraph(article_id, paragraph_id)
                    if self.widgets['default_saving_article'].value is True:
                        self.article_saver.add_article(article_id)
                    chk_article, chk_paragraph = self._create_saving_checkboxes(article_id, paragraph_id)

                display(HTML(article_metadata))
                if self.article_saver:
                    # display(radio_button)
                    display(chk_paragraph)
                    display(chk_article)
                display(HTML(formatted_output))

                print()
                self.report += article_metadata + formatted_output + '<br>'

    def _on_save_paragraph_change(self, change, article_id=None, paragraph_id=None):
        with self.widgets['out']:
            print(f"Toggled: {article_id}, {paragraph_id}")
            print(f"Change: {change}")
        if change["new"] is True:
            self.article_saver.add_paragraph(article_id, paragraph_id)
        else:
            self.article_saver.remove_paragraph(article_id, paragraph_id)

    def _on_save_article_change(self, change, article_id=None):
        with self.widgets['out']:
            print(f"Toggled: {article_id}")
            print(f"Change: {change}")
        if change["new"] is True:
            self.article_saver.add_article(article_id)
        else:
            self.article_saver.remove_article(article_id)

    def _create_saving_checkboxes(self, article_id, paragraph_id):
        chk_paragraph = widgets.Checkbox(
            value=False,
            description='Save Paragraph',
            indent=False,
            disabled=False,
        )
        chk_article = widgets.Checkbox(
            value=False,
            description='Save Article',
            indent=False,
            disabled=False,
        )

        chk_paragraph.observe(
            handler=functools.partial(
                self._on_save_paragraph_change,
                article_id=article_id,
                paragraph_id=paragraph_id),
            names="value")
        chk_article.observe(
            handler=functools.partial(
                self._on_save_article_change,
                article_id=article_id),
            names="value")

        if self.article_saver is None:
            chk_paragraph.disabled = True
            chk_article.disabled = True
        else:
            chk_paragraph.value = self.article_saver.has_paragraph(article_id, paragraph_id)
            chk_article.value = self.article_saver.has_article(article_id)

        return chk_article, chk_paragraph

    def status_article_retrieve(self, article_infos):
        """Return information about the saving choice of this article."""
        color_text = '#bdbdbd'
        status = self.article_saver.status_on_article_retrieve(article_infos)
        status = f"""<p style="font-size:13px; color:{color_text}"> {status} </p>"""
        return status

    def create_radio_buttons(self, article_infos, articles_metadata):
        """Create radio button."""
        default_value = self.article_saver.saved_articles[article_infos] \
            if article_infos in self.article_saver.saved_articles.keys() \
            else self.widgets['default_value_article_saver'].value

        def on_value_change(change):
            for infos, button in self.radio_buttons:
                self.article_saver.saved_articles[infos] = button.value
            return change['new']

        radio_button = widgets.ToggleButtons(
            options=self.saving_options,
            value=default_value,
            description='Saving: ',
            style={'description_width': 'initial', 'button_width': '200px'},
            disabled=False)
        radio_button.observe(on_value_change, names='value')

        if radio_button.value != SAVING_OPTIONS['nothing']:
            self.article_saver.saved_articles[article_infos] = radio_button.value

        self.article_saver.articles_metadata[article_infos[0]] = articles_metadata
        self.radio_buttons.append((article_infos, radio_button))

        return radio_button

    def article_report_on_click(self, change_dict):
        """Create the saved articles report."""
        with self.widgets['out']:
            self.article_saver.report()
            print('The PDF report has been created.')

    def report_on_click(self, change_dict):
        """Create the report of the search."""
        with self.widgets['out']:
            print('The PDF report has been created.')

        color_hyperparameters = '#222222'

        hyperparameters_section = f"""
        <h1> Search Parameters </h1>
        <ul style="font-size:13; color:{color_hyperparameters}">
        <li> {'</li> <li>'.join([
            '<b>' +
            ' '.join(k.split('_')).title() +
            '</b>' +
            f': {repr(v.value)}'
            for k, v in self.widgets.items()
            if hasattr(v, 'value')])}
        </li>
        </ul>
        """

        results_section = f"<h1> Results </h1> {self.report}"
        pdfkit.from_string(hyperparameters_section + results_section,
                           f"report_{datetime.datetime.now()}.pdf")

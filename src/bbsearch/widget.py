"""Module for the widget."""
from collections import OrderedDict
import datetime
import logging
import pdfkit
import textwrap

import ipywidgets as widgets
import IPython
from IPython.display import HTML

from .search import run_search
from .sql import find_paragraph

logger = logging.getLogger(__name__)


class Widget:
    """Widget for search engine.

    Parameters
    ----------
    embedding_models: dict
        Dictionary containing instances of the different embedding models.
        The keys have to be the name of the different models and
        the values have to be instance of 'EmbeddingModel' class.

    precomputed_embeddings: dict
        Dictionary containing the precomputed embeddings.
        The keys have to be the name of the differents models and
        the values have to be numpy arrays containing the embeddings.

    database: sqlite3.Cursor
        Cursor to the database used for the search engine.
    """

    def __init__(self,
                 embedding_models,
                 precomputed_embeddings,
                 database):

        self.embedding_models = embedding_models
        self.precomputed_embeddings = precomputed_embeddings
        self.database = database

        self.report = ''

        self.my_widgets = OrderedDict()
        self.initialize_widgets()

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
        """Retrieve metadata and complete the report with HTML string given an sentence_id.

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
        """
        article_sha, section_name, text = \
            self.database.execute(
                'SELECT sha, section_name, text FROM sentences WHERE sentence_id = ?',
                [sentence_id]).fetchall()[0]
        (article_id,) = self.database.execute(
            'SELECT article_id FROM article_id_2_sha WHERE sha = ?',
            [article_sha]).fetchall()[0]
        article_auth, article_title, date, ref = self.database.execute(
            'SELECT authors, title, date, url FROM articles WHERE article_id = ?',
            [article_id]).fetchall()[0]
        article_auth = article_auth.split(';')[0] + ' et al.'
        ref = ref if ref else ''
        section_name = section_name if section_name else ''

        width = 80
        if print_whole_paragraph:
            try:
                paragraph = find_paragraph(sentence_id, self.database)
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

        return article_metadata, formatted_output

    def initialize_widgets(self):
        """Initialize widget dictionary."""
        # Select model to compute Sentence Embeddings
        self.my_widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['USE', 'SBERT', 'BSV', 'SBIOBERT'],
            description='Model for Sentence Embedding',
            tooltips=['Universal Sentence Encoder', 'Sentence BERT', 'BioSentVec',
                      'Sentence BioBERT'], )

        # Select n. of top results to return
        self.my_widgets['top_results'] = widgets.widgets.IntSlider(
            value=10,
            min=0,
            max=100,
            description='Top N results')

        # Choose whether to print whole paragraph containing sentence highlighted, or just the
        # sentence
        self.my_widgets['print_paragraph'] = widgets.Checkbox(
            value=True,
            description='Show whole paragraph')

        # Enter Query
        self.my_widgets['query_text'] = widgets.Textarea(
            value='Glucose is a risk factor for COVID-19',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Query')

        self.my_widgets['has_journal'] = widgets.Checkbox(
            description="Require Journal",
            value=False)

        self.my_widgets['date_range'] = widgets.IntRangeSlider(
            description="Date Range:",
            continuous_update=False,
            min=1900,
            max=2020,
            value=(1900, 2020),
            layout=widgets.Layout(width='80ch'))

        # Enter Deprioritization Query
        self.my_widgets['deprioritize_text'] = widgets.Textarea(
            value='',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Deprioritize')

        # Select Deprioritization Strength
        self.my_widgets['deprioritize_strength'] = widgets.ToggleButtons(
            options=['None', 'Weak', 'Mild', 'Strong', 'Stronger'],
            disabled=False,
            button_style='info',
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength', )

        # Enter Substrings Exclusions
        self.my_widgets['exclusion_text'] = widgets.Textarea(
            layout=widgets.Layout(width='90%', height='80px'),
            value='',
            style={'description_width': 'initial'},
            description='Substring Exclusion (newline separated): ')

        # Click to run Information Retrieval!
        self.my_widgets['investigate_button'] = widgets.Button(description='Investigate!')

        # Click to run Generate Report!
        self.my_widgets['report_button'] = widgets.Button(description='Generate PDF Report!',
                                                          layout=widgets.Layout(width='25%'))

        # Output Area
        self.my_widgets['out'] = widgets.Output(layout={'border': '1px solid black'})

        # Callbacks
        self.my_widgets['investigate_button'].on_click(self.investigate_on_click)
        self.my_widgets['report_button'].on_click(self.report_on_click)

    def hide_from_user(self):
        """Hide from the user not used functionalities in the widgets."""
        self.my_widgets['exclusion_text'].layout.display = 'none'
        # Remove some models (USE and SBERT)
        self.my_widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['BSV', 'SBIOBERT'],
            description='Model for Sentence Embedding',
            tooltips=['BioSentVec', 'Sentence BioBERT'], )
        # Remove some deprioritization strength
        self.my_widgets['deprioritize_strength'] = widgets.ToggleButtons(
            options=['None', 'Mild', 'Stronger'],
            disabled=False,
            button_style='info',
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength')

    def investigate_on_click(self, change_dict):
        """Investigate button."""
        self.my_widgets['out'].clear_output()
        with self.my_widgets['out']:
            sentence_embedder_name = self.my_widgets['sent_embedder'].value
            k = self.my_widgets['top_results'].value
            print_whole_paragraph = self.my_widgets['print_paragraph'].value
            query_text = self.my_widgets['query_text'].value
            deprioritize_text = self.my_widgets['deprioritize_text'].value
            deprioritize_strength = self.my_widgets['deprioritize_strength'].value
            exclusion_text = self.my_widgets['exclusion_text'].value \
                if 'exclusion_text' in self.my_widgets.keys() else ''
            has_journal = self.my_widgets['has_journal'].value
            date_range = self.my_widgets['date_range'].value

            sentence_ids, _, _ = run_search(self.embedding_models[sentence_embedder_name],
                                            self.precomputed_embeddings[sentence_embedder_name],
                                            database=self.database,
                                            k=k,
                                            query_text=query_text,
                                            has_journal=has_journal,
                                            date_range=date_range,
                                            deprioritize_strength=deprioritize_strength,
                                            deprioritize_text=deprioritize_text,
                                            exclusion_text=exclusion_text)

            print(f'\nInvestigating: {query_text}\n')

            for sentence_id in sentence_ids:
                article_metadata, formatted_output = self.print_single_result(sentence_id, print_whole_paragraph)

                IPython.display.display(HTML(article_metadata))
                IPython.display.display(HTML(formatted_output))

                print()
                self.report += article_metadata + formatted_output + '<br>'

    def report_on_click(self, change_dict):
        """Create the report of the search."""
        print("Saving results to a pdf file.")

        color_hyperparameters = '#222222'

        hyperparameters_section = f"""
        <h1> Search Parameters </h1>
        <ul style="font-size:13; color:{color_hyperparameters}">
        <li> {'</li> <li>'.join([
            '<b>' +
            ' '.join(k.split('_')).title() +
            '</b>' +
            f': {repr(v.value)}'
            for k, v in self.my_widgets.items()
            if hasattr(v, 'value')])}
        </li>
        </ul>
        """

        results_section = f"<h1> Results </h1> {self.report}"
        pdfkit.from_string(hyperparameters_section + results_section,
                           f"report_{datetime.datetime.now()}.pdf")

    def display(self):
        """Display the widget."""
        ordered_widgets = list(self.my_widgets.values())
        main_widget = widgets.VBox(ordered_widgets)
        IPython.display.display(main_widget)

"""Module for the widget."""
from collections import OrderedDict
import datetime
import logging
import pdfkit
import textwrap

import ipywidgets as widgets
from IPython.display import display, HTML

from .sql import find_paragraph

logger = logging.getLogger(__name__)

SAVING_OPTIONS = OrderedDict([('nothing',  'Do not take this article'),
                              ('paragraph', 'Extract the paragraph'),
                              ('article', 'Extract the entire article')])


class Widget:
    """Widget for search engine.

    Parameters
    ----------
    searcher : bbsearch.search.LocalSearcher or bbsearch.remote_searcher.RemoteSearcher
        The search engine.

    article_saver: ArticleSaver
        If specified, this article saver will keep all the article_id
        of interest for the user during the different queries.
    """

    def __init__(self,
                 searcher,
                 database,
                 article_saver=None):

        self.searcher = searcher
        self.database = database

        self.report = ''

        self.radio_buttons = list()
        self.article_saver = article_saver
        self.saving_options = list(SAVING_OPTIONS.values())

        self.my_widgets = OrderedDict()
        self.initialize_widgets()
        self.hide_from_user()

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
        article_infos : tuple
            A tuple with two elements (article_id, paragraph_id) containing
            the information about the article.

        """
        article_sha, section_name, text, paragraph_id = \
            self.database.execute(
                'SELECT sha, section_name, text, paragraph_id FROM sentences WHERE sentence_id = ?',
                [sentence_id]).fetchall()[0]
        (article_id,) = self.database.execute(
            'SELECT article_id FROM article_id_2_sha WHERE sha = ?',
            [article_sha]).fetchall()[0]
        article_auth, article_title, date, ref = self.database.execute(
            'SELECT authors, title, date, url FROM articles WHERE article_id = ?',
            [article_id]).fetchall()[0]
        try:
            article_auth = article_auth.split(';')[0] + ' et al.'
        except AttributeError:
            article_auth = ''

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
        article_metadata = textwrap.dedent(article_metadata)

        article_infos = (article_id, paragraph_id)

        return article_metadata, formatted_output, article_infos

    def initialize_widgets(self):
        """Initialize widget dictionary."""
        # Select model to compute Sentence Embeddings
        self.my_widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['USE', 'SBERT', 'BSV', 'SBioBERT'],
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

        self.my_widgets['default_value_article_saver'] = widgets.ToggleButtons(
            options=self.saving_options,
            disabled=False,
            style={'description_width': 'initial', 'button_width': '200px'},
            description='Default saving: ', )

        def default_value_article_saver_change(change):
            if self.radio_buttons:
                for article_infos, button in self.radio_buttons:
                    button.value = change['new']
                return change['new']

        self.my_widgets['default_value_article_saver'].observe(default_value_article_saver_change,
                                                               names='value')

        # Click to run Information Retrieval!
        self.my_widgets['investigate_button'] = widgets.Button(description='Investigate!',
                                                               layout=widgets.Layout(width='50%'))

        # Click to run Generate Report!
        self.my_widgets['report_button'] = widgets.Button(description='Generate Report of Search Results',
                                                          layout=widgets.Layout(width='50%'))

        self.my_widgets['articles_button'] = widgets.Button(description='Generate Report of Selected Articles',
                                                            layout=widgets.Layout(width='50%'))
        # Output Area
        self.my_widgets['out'] = widgets.Output(layout={'border': '1px solid black'})

        # Callbacks
        self.my_widgets['investigate_button'].on_click(self.investigate_on_click)
        self.my_widgets['report_button'].on_click(self.report_on_click)
        self.my_widgets['articles_button'].on_click(self.article_report_on_click)

    def hide_from_user(self):
        """Hide from the user not used functionalities in the widgets."""
        self.my_widgets['exclusion_text'].layout.display = 'none'
        # Remove some models (USE and SBERT)
        self.my_widgets['sent_embedder'] = widgets.ToggleButtons(
            options=['BSV', 'SBioBERT'],
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
        # Clear widget output + All the radio buttons
        self.my_widgets['out'].clear_output()
        self.radio_buttons = list()

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

            sentence_ids, _, _ = self.searcher.query(
                sentence_embedder_name,
                k=k,
                query_text=query_text,
                has_journal=has_journal,
                date_range=date_range,
                deprioritize_strength=deprioritize_strength,
                deprioritize_text=deprioritize_text,
                exclusion_text=exclusion_text)

            print(f'\nInvestigating: {query_text}\n')

            for sentence_id in sentence_ids:
                if self.article_saver:
                    article_metadata, formatted_output, article_infos = \
                        self.print_single_result(int(sentence_id), print_whole_paragraph)

                    radio_button = self.create_radio_buttons(article_infos, article_metadata)
                    status = self.status_article_retrieve(article_infos)

                display(HTML(article_metadata))
                if self.article_saver:
                    display(radio_button)
                    display(HTML(status))
                display(HTML(formatted_output))

                print()
                self.report += article_metadata + formatted_output + '<br>'

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
            else self.my_widgets['default_value_article_saver'].value

        radio_button = widgets.ToggleButtons(
            options=self.saving_options,
            value=default_value,
            description='Saving: ',
            style={'description_width': 'initial', 'button_width': '200px'},
            disabled=False)

        if radio_button.value != SAVING_OPTIONS['article']:
            self.article_saver.saved_articles[article_infos] = radio_button.value

        self.article_saver.articles_metadata[article_infos[0]] = articles_metadata

        def on_value_change(change):
            for infos, button in self.radio_buttons:
                self.article_saver.saved_articles[infos] = button.value
            return change['new']

        self.radio_buttons.append((article_infos, radio_button))
        self.radio_buttons[-1][-1].observe(on_value_change, names='value')
        return radio_button

    def article_report_on_click(self, change_dict):
        """Create the saved articles report."""
        with self.my_widgets['out']:
            self.article_saver.report()
            print('The PDF report has been created.')

    def report_on_click(self, change_dict):
        """Create the report of the search."""
        with self.my_widgets['out']:
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
        display(main_widget)

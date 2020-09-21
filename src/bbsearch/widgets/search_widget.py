"""The Search widget."""
import datetime
import enum
import functools
import logging
import math
import textwrap

import ipywidgets as widgets
import pdfkit
from IPython.display import HTML, display

from .._css import style
from ..sql import (
    retrieve_article_metadata_from_article_id,
    retrieve_paragraph_from_sentence_id,
    retrieve_sentences_from_sentence_ids,
)
from ..utils import Timer

logger = logging.getLogger(__name__)


class _Save(enum.Enum):
    NOTHING = enum.auto()
    PARAGRAPH = enum.auto()
    ARTICLE = enum.auto()


class SearchWidget(widgets.VBox):
    """Widget for search engine.

    Parameters
    ----------
    searcher : bbsearch.search.LocalSearcher or
               bbsearch.remote_searcher.RemoteSearcher
        The search engine.

    connection : SQLAlchemy connectable (engine/connection) or database str URI or DBAPI2 connection (fallback mode)
        Connection to the SQL database

    article_saver: bbsearch.widgets.ArticleSaver, optional
        If specified, this article saver will keep all the article_id
        of interest for the user during the different queries.

    results_per_page : int, optional
        The number of results to display per results page.
    """

    def __init__(self,
                 searcher,
                 connection,
                 article_saver=None,
                 results_per_page=10
                 ):
        super().__init__()

        self.searcher = searcher
        self.connection = connection
        self.article_saver = article_saver
        self.results_per_page = max(1, results_per_page)
        self.n_pages = 1
        self.current_page = -1

        self.saving_labels = {
            _Save.NOTHING: 'Do not take this article',
            _Save.PARAGRAPH: 'Extract the paragraph',
            _Save.ARTICLE: 'Extract the entire article'}

        self.radio_buttons = []
        self.current_sentence_ids = []
        self.current_paragraph_ids = []
        self.current_article_ids = []

        self.widgets_style = {'description_width': 'initial'}
        self.widgets = dict()
        self._init_widgets()
        self._adjust_widgets()
        self._init_ui()

    def _init_widgets(self):
        """Initialize widget dictionary."""
        # Select model to compute Sentence Embeddings
        self.widgets['sent_embedder'] = widgets.RadioButtons(
            options=['USE', 'SBERT', 'BSV', 'SBioBERT', 'Sent2Vec'],
            description='Model for Sentence Embedding',
            tooltips=['Universal Sentence Encoder', 'Sentence BERT', 'BioSentVec',
                      'Sentence BioBERT', 'Sent2Vec Model'],
            style=self.widgets_style
            )

        # Select n. of top results to return
        self.widgets['top_results'] = widgets.widgets.IntText(
            value=20,
            description='Top N results',
            style=self.widgets_style
            )

        # Choose whether to print whole paragraph containing sentence highlighted, or just the
        # sentence
        self.widgets['print_paragraph'] = widgets.Checkbox(
            value=True,
            description='Show whole paragraph',
            style=self.widgets_style
            )

        # Enter Query
        self.widgets['query_text'] = widgets.Textarea(
            value='Glucose is a risk factor for COVID-19',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Query',
            style=self.widgets_style
            )

        self.widgets['has_journal'] = widgets.Checkbox(
            description="Only articles from journals",
            value=True,
            style=self.widgets_style
            )

        self.widgets['date_range'] = widgets.IntRangeSlider(
            description="Date Range:",
            continuous_update=False,
            min=1850,
            max=2020,
            value=(2000, 2020),
            layout=widgets.Layout(width='80ch'),
            style=self.widgets_style
            )
        # Enter Deprioritization Query
        self.widgets['deprioritize_text'] = widgets.Textarea(
            value='',
            layout=widgets.Layout(width='90%', height='80px'),
            description='Deprioritize',
            style=self.widgets_style
            )

        # Select Deprioritization Strength
        self.widgets['deprioritize_strength'] = widgets.RadioButtons(
            options=['None', 'Weak', 'Mild', 'Strong', 'Stronger'],
            disabled=False,
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength',
            )

        # Enter Substrings Exclusions
        self.widgets['exclusion_text'] = widgets.Textarea(
            layout=widgets.Layout(width='90%', height='80px'),
            value='',
            style=self.widgets_style,
            description='Substring Exclusion (newline separated): '
            )

        self.widgets['inclusion_text'] = widgets.Textarea(
            layout=widgets.Layout(width='90%'),
            value='',
            style=self.widgets_style,
            description="Exact phrase matching:",
            rows=5,
            placeholder=textwrap.dedent("""
                    Case insensitive,  one phrase per line. Valid phrases are:
                    1. Single word                      : glucose
                    2. Multiple words                   : risk factor
                    3. Single word with variable suffix : molecul*
                       (matches "molecule", "molecules", "molecular")
                    """).strip(),
        )

        self.widgets['default_value_article_saver'] = widgets.RadioButtons(
            options=[
                (self.saving_labels[_Save.NOTHING], _Save.NOTHING),
                (self.saving_labels[_Save.PARAGRAPH], _Save.PARAGRAPH),
                (self.saving_labels[_Save.ARTICLE], _Save.ARTICLE)],
            value=_Save.ARTICLE,
            disabled=False,
            style={'description_width': 'initial', 'button_width': '200px'},
            description='Default saving: '
            )

        # Click to run Information Retrieval!
        self.widgets['investigate_button'] = widgets.Button(
            description='📚 Search Literature!',
            layout=widgets.Layout(width='350px', height='50px'))
        self.widgets['investigate_button'].add_class('bbs_button')

        # Click to run Generate Report!
        self.widgets['report_button'] = widgets.Button(
            description='Generate Report of Search Results',
            layout=widgets.Layout(width='50%'))

        self.widgets['articles_button'] = widgets.Button(
            description='Generate Report of Selected Articles',
            layout=widgets.Layout(width='50%'))

        # Output Area
        self.widgets['out'] = widgets.Output(
            layout={'border': '1px solid black'})

        # Status Area
        self.widgets['status'] = widgets.Output(
            layout={'border': '1px solid black', 'flex': '1'})
        self.widgets['status_clear'] = widgets.Button(
            description="Clear",
            layout={'max_width': '100px'})
        self.widgets['status_clear'].on_click(
            lambda b: self.widgets['status'].clear_output())

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
        self.widgets['investigate_button'].on_click(self._cb_bt_investigate)
        self.widgets['report_button'].on_click(self._cb_bt_pdf_report_search)
        self.widgets['articles_button'].on_click(self._cb_bt_pdf_report_article_saver)

    def _adjust_widgets(self):
        """Hide from the user not used functionalities in the widgets."""
        self.widgets['exclusion_text'].layout.display = 'none'
        # Remove some models: (USE, SBERT, SBioBERT)
        self.widgets['sent_embedder'] = widgets.RadioButtons(
            options=['BSV', 'Sent2Vec'],
            description='Model for Sentence Embedding',
            tooltips=['BioSentVec Model', 'Sent2Vec Model'],
            style=self.widgets_style)
        # Remove some deprioritization strength
        self.widgets['deprioritize_strength'] = widgets.RadioButtons(
            options=['None', 'Mild', 'Stronger'],
            disabled=False,
            style={'description_width': 'initial', 'button_width': '80px'},
            description='Deprioritization strength')

    def _init_ui(self):
        css_style = style.get_css_style()
        display(HTML(f'<style> {css_style} </style>'))

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
            self.widgets['inclusion_text'],
            self.widgets['default_value_article_saver'],
            self.widgets['investigate_button'],
            page_selection,
            self.widgets['out'],
            page_selection,
            widgets.HBox(children=(
                self.widgets['status'],
                self.widgets['status_clear'])),
            self.widgets['report_button'],
            self.widgets['articles_button'],
        ]

        with self.widgets['status']:
            init_text = r"""
              ____  ____   _____
             |  _ \|  _ \ / ____|
             | |_) | |_) | (___
             |  _ <|  _ < \___ \
             | |_) | |_) |____) |
             |____/|____/|_____/

            Click on "Search Literature!" button to display some results.
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
        start = paragraph.index(sentence)
        end = start + len(sentence)
        highlighted_paragraph = \
            f"""
            <div class="paragraph">
                {paragraph[:start]} <div class="paragraph_emph"> {paragraph[start:end]} </div> {paragraph[end:]}
            </div>
            """

        return highlighted_paragraph

    def _fetch_result_info(self, sentence_id):
        """Fetch information for a sentence ID from the database.

        Parameters
        ----------
        sentence_id : int
            The sentence_id for a search result.

        Returns
        -------
        result_info : dict
            A dictionary containing the following fields:

                "sentence_id"
                "paragraph_id"
                "article_id"
                "article_title"
                "article_auth"
                "ref"
                "section_name"
                "text"
        """
        sentence = retrieve_sentences_from_sentence_ids(sentence_ids=(sentence_id,),
                                                        engine=self.connection)
        article_id, section_name, text, paragraph_id = \
            sentence.iloc[0][['article_id', 'section_name',
                              'text', 'paragraph_pos_in_article']]

        article = retrieve_article_metadata_from_article_id(article_id=article_id,
                                                            engine=self.connection)
        article_auth, article_title, ref = \
            article.iloc[0][['authors', 'title', 'url']]

        try:
            article_auth = article_auth.split(';')[0] + ' et al.'
        except AttributeError:
            article_auth = ''

        ref = ref or ""
        section_name = section_name or ""

        result_info = {
            "sentence_id": sentence_id,
            "paragraph_id": int(paragraph_id),
            "article_id": article_id,
            "article_title": article_title,
            "article_auth": article_auth,
            "ref": ref,
            "section_name": section_name,
            "text": text
        }

        return result_info

    def print_single_result(self, result_info, print_whole_paragraph):
        """Retrieve metadata and complete the report with HTML string given sentence_id.

        Parameters
        ----------
        result_info: dict
            The information for a single result obtained by calling
            `_fetch_result_info`.

        print_whole_paragraph: bool
            If true, the whole paragraph will be displayed in the results of the widget.

        Returns
        -------
        article_metadata: str
            Formatted string containing the metadata of the article.
        formatted_output: str
            Formatted output of the sentence.
        """
        sentence_id = result_info["sentence_id"]
        text = result_info["text"]
        ref = result_info["ref"]
        article_title = result_info["article_title"]
        article_auth = result_info["article_auth"]
        section_name = result_info["section_name"]

        width = 80
        if print_whole_paragraph:
            try:
                paragraph = retrieve_paragraph_from_sentence_id(sentence_id,
                                                                self.connection)
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

        # color_title = '#1A0DAB'
        # color_metadata = '#006621'
        article_metadata = f"""
            <a href="{ref}">
                <div class="article_title">
                    {article_title}
                </div>
            </a>
            <div class="metadata">
                {article_auth} &#183; {section_name.lower().title()}
            </div>
            """
        article_metadata = textwrap.dedent(article_metadata)

        return article_metadata, formatted_output

    def _cb_bt_investigate(self, change_dict):
        """Investigate button callback."""
        # Get user selection
        which_model = self.widgets['sent_embedder'].value
        k = self.widgets['top_results'].value
        query_text = self.widgets['query_text'].value
        deprioritize_text = self.widgets['deprioritize_text'].value
        deprioritize_strength = self.widgets['deprioritize_strength'].value
        exclusion_text = self.widgets['exclusion_text'].value \
            if 'exclusion_text' in self.widgets.keys() else ''
        inclusion_text = self.widgets['inclusion_text'].value
        has_journal = self.widgets['has_journal'].value
        date_range = self.widgets['date_range'].value

        # Clear output and show waiting message
        timer = Timer()
        self.widgets['out'].clear_output()
        self.widgets['status'].clear_output()
        with self.widgets['status']:
            header = f'Query: \"{query_text}\"'
            print(header)
            print('-' * len(header))

            print('Sending query to server...'.ljust(50), end='', flush=True)
            with timer("server query"):
                self.current_sentence_ids, *_ = self.searcher.query(
                    which_model=which_model,
                    k=k,
                    query_text=query_text,
                    has_journal=has_journal,
                    date_range=date_range,
                    deprioritize_strength=deprioritize_strength,
                    deprioritize_text=deprioritize_text,
                    exclusion_text=exclusion_text,
                    inclusion_text=inclusion_text)
            print(f'{timer["server query"]:7.2f} seconds')

            print('Resolving articles...'.ljust(50), end='', flush=True)
            with timer("id resolution"):
                self.current_article_ids, self.current_paragraph_ids = \
                    self.resolve_ids(self.current_sentence_ids)
            print(f'{timer["id resolution"]:7.2f} seconds')

            print('Applying default saving...'.ljust(50), end='', flush=True)
            with timer("default saving"):
                self._apply_default_saving()
            print(f'{timer["default saving"]:7.2f} seconds')

            print('Updating the results display...'.ljust(50), end='', flush=True)
            with timer("update page"):
                self.n_pages = math.ceil(
                    len(self.current_sentence_ids) / self.results_per_page)
                self.set_page(0, force=True)
            print(f'{timer["update page"]:7.2f} seconds')

            print('Done.')

    def _apply_default_saving(self):
        default_saving_value = self.widgets["default_value_article_saver"].value
        if default_saving_value != _Save.NOTHING:
            for article_id, paragraph_id in zip(self.current_article_ids, self.current_paragraph_ids):
                if default_saving_value == _Save.ARTICLE:
                    self.article_saver.add_article(article_id)
                elif default_saving_value == _Save.PARAGRAPH:
                    self.article_saver.add_paragraph(article_id, paragraph_id)

    def resolve_ids(self, sentence_ids):
        """Resolve sentence IDs into article and paragraph IDs.

        Parameters
        ----------
        sentence_ids : list_like
            A list of sentence IDs to be resolved

        Returns
        -------
        article_ids : list_like
            The article IDs corresponding to the sentence IDs
        paragraph_ids : list_like
            The paragraph IDs corresponding to the sentence IDs
        """
        sentences = retrieve_sentences_from_sentence_ids(sentence_ids=sentence_ids,
                                                         engine=self.connection)
        article_ids = sentences['article_id'].to_list()
        paragraph_ids = sentences['paragraph_pos_in_article'].to_list()

        return article_ids, paragraph_ids

    def set_page(self, new_page, force=False):
        """Go to a given page in the results view.

        Parameters
        ----------
        new_page : int
            The new page number to go to.
        force : bool
            By default, if `new_page` is the same one as the one
            currently viewed, the the page is not reloaded. To reload
            the page set this parameter to True. This is ueful when
            new results have been fetched and so the view needs to
            be updated.
        """
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
                result_info = self._fetch_result_info(sentence_id)
                article_metadata, formatted_output = \
                    self.print_single_result(result_info, print_whole_paragraph)
                if self.article_saver:
                    chk_article, chk_paragraph = self._create_saving_checkboxes(
                        result_info["article_id"],
                        result_info["paragraph_id"])

                display(HTML(article_metadata))
                if self.article_saver:
                    # display(radio_button)
                    display(chk_paragraph)
                    display(chk_article)
                display(HTML(formatted_output))

                print()

    def _cb_chkb_save_paragraph(self, change, article_id=None, paragraph_id=None):
        if change["new"] is True:
            self.article_saver.add_paragraph(article_id, paragraph_id)
        else:
            self.article_saver.remove_paragraph(article_id, paragraph_id)

    def _cb_chkb_save_article(self, change, article_id=None):
        if change["new"] is True:
            self.article_saver.add_article(article_id)
        else:
            self.article_saver.remove_article(article_id)

    def _create_saving_checkboxes(self, article_id, paragraph_id):
        chk_paragraph = widgets.Checkbox(
            value=False,
            description=self.saving_labels[_Save.PARAGRAPH],
            indent=False,
            disabled=False,
        )
        chk_article = widgets.Checkbox(
            value=False,
            description=self.saving_labels[_Save.ARTICLE],
            indent=False,
            disabled=False,
        )

        chk_paragraph.observe(
            handler=functools.partial(
                self._cb_chkb_save_paragraph,
                article_id=article_id,
                paragraph_id=paragraph_id),
            names="value")
        chk_article.observe(
            handler=functools.partial(
                self._cb_chkb_save_article,
                article_id=article_id),
            names="value")

        if self.article_saver is None:
            chk_paragraph.disabled = True
            chk_article.disabled = True
        else:
            # Check if this article/paragraph has been saved before
            if self.article_saver.has_paragraph(article_id, paragraph_id):
                chk_paragraph.value = True
            if self.article_saver.has_article(article_id):
                chk_article.value = True

        return chk_article, chk_paragraph

    def _cb_bt_pdf_report_article_saver(self, change_dict):
        """Create the saved articles report."""
        with self.widgets['status']:
            print()
            print('Creating the saved results PDF report... ')
            self.article_saver.pdf_report()

    def _cb_bt_pdf_report_search(self, change_dict):
        """Create the report of the search."""
        with self.widgets['status']:
            print()
            print('Creating the search results PDF report... ')

            hyperparameters_section = f"""
            <h1> Search Parameters </h1>
            <ul class="paragraph">
            <li> {'</li> <li>'.join([
                '<div class="paragraph_emph">' +
                ' '.join(k.split('_')).title() +
                '</b>' +
                f': {repr(v.value)}'
                for k, v in self.widgets.items()
                if hasattr(v, 'value')])}
            </li>
            </ul>
            """

            print_whole_paragraph = self.widgets['print_paragraph'].value
            report = ""
            for sentence_id in self.current_sentence_ids:
                result_info = self._fetch_result_info(sentence_id)
                article_metadata, formatted_output = \
                    self.print_single_result(result_info, print_whole_paragraph)
                report += article_metadata + formatted_output + '<br>'

            results_section = f"<h1> Results </h1> {report}"

            css_style = style.get_css_style()

            pdfkit.from_string(f"<style> {css_style} </style>" +
                               hyperparameters_section +
                               results_section,
                               f"report_{datetime.datetime.now()}.pdf")

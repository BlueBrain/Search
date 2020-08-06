"""Module for the mining widget."""
from dataclasses import dataclass
import io

from IPython.display import display, HTML
import ipywidgets as widgets
import pandas as pd
import requests

from .._css import style
from ..utils import Timer


@dataclass
class SchemaRequest:
    """Class for keeping track of request schema in a mutable way."""

    schema: pd.DataFrame = pd.DataFrame()


class MiningWidget(widgets.VBox):
    """The mining widget.

    Parameters
    ----------
    mining_server_url : str
        The URL of the mining server.
    schema_request : SchemaRequest
        An object holding a dataframe with the requested mining schema (entity, relation, attribute types).
    article_saver : bbsearch.widgets.ArticleSaver
        An instance of the article saver.
    default_text : string, optional
        The default text assign to the text area.
    """

    def __init__(self, mining_server_url, schema_request, article_saver=None, default_text=''):
        super().__init__()

        self.mining_server_url = mining_server_url
        self.article_saver = article_saver
        self.schema_request = schema_request

        # This is the output: csv table of extracted entities/relations.
        self.table_extractions = None

        # Define Widgets
        self.widgets = dict()

        self._init_widgets(default_text)
        self._init_ui()

    def _init_widgets(self, default_text):
        # "Input Text" Widget
        self.widgets['input_text'] = widgets.Textarea(
            value=default_text,
            layout=widgets.Layout(width='75%', height='300px'))

        # "Mine This Text" button
        self.widgets['mine_text'] = widgets.Button(
            description='⚒️  Mine This Text!',
            layout=widgets.Layout(width='350px', height='50px'))
        self.widgets['mine_text'].on_click(self._mine_text_clicked)
        self.widgets['mine_text'].add_class('bbs_button')

        # "Mine Selected Articles" button
        self.widgets['mine_articles'] = widgets.Button(
            description='⚒️  Mine Selected Articles!',
            layout=widgets.Layout(width='350px', height='50px'))
        self.widgets['mine_articles'].on_click(self._mine_articles_clicked)
        self.widgets['mine_articles'].add_class('bbs_button')

        # "Output Area" Widget
        self.widgets['out'] = widgets.Output(layout={'border': '0.5px solid black'})

    def _init_ui(self):
        css_style = style.get_css_style()
        display(HTML(f'<style> {css_style} </style>'))

        self.children = [
            self.widgets['input_text'],
            widgets.HBox(children=[
                self.widgets['mine_text'],
                self.widgets['mine_articles']]),
            self.widgets['out'],
        ]

    def textmining_pipeline(self, information, schema_df, debug=False):
        """Handle text mining server requests depending on the type of information.

        Parameters
        ----------
        information: str or list.
            Information can be either a raw string text, either a list of tuples
            (article_id, paragraph_id) related to the database.
        schema_df : pd.DataFrame
            A dataframe with the requested mining schema (entity, relation, attribute types).
        debug : bool
            If True, columns are not necessarily matching the specification. However, they
            contain debugging information. If False, then matching exactly the specification.

        Returns
        -------
        table_extractions: pd.DataFrame
            The final table. If `debug=True` then it contains all the metadata. If False then it
            only contains columns in the official specification.
        """
        schema_str = schema_df.to_csv(path_or_buf=None, index=False)
        if isinstance(information, list):
            response = requests.post(
                self.mining_server_url + '/database',
                json={"identifiers": information,
                      "schema": schema_str,
                      }
            )
        elif isinstance(information, str):
            response = requests.post(
                self.mining_server_url + '/text',
                json={"text": information,
                      "schema": schema_str,
                      "debug": debug
                      }
            )
        else:
            raise TypeError('Wrong type for the information!')

        table_extractions = None
        if response.status_code == 200:
            response_dict = response.json()
            for warning_msg in response_dict['warnings']:
                display(HTML(f'<div style="color:#BA4A00"> <b>WARNING!</b> {warning_msg} </div>'))
            with io.StringIO(response_dict['csv_extractions']) as f:
                table_extractions = pd.read_csv(f)
        else:
            print("Server response is ERROR!")
            print(response.headers)
            print(response.text)

        return table_extractions

    def _mine_articles_clicked(self, b):
        self.widgets['out'].clear_output()

        if self.article_saver is None:
            with self.widgets['out']:
                print("No article saver was provided. Nothing to mine.")
            return

        with self.widgets['out']:
            timer = Timer()

            print("Collecting saved items...".ljust(50), end='', flush=True)
            with timer("collect items"):
                identifiers = self.article_saver.get_saved_items()

            print(f'{timer["collect items"]:7.2f} seconds')
            print('Mining request schema:')
            display(self.schema_request.schema)
            print("Running the mining pipeline...".ljust(50), end='', flush=True)
            with timer("pipeline"):
                self.table_extractions = self.textmining_pipeline(
                    information=identifiers,
                    schema_df=self.schema_request.schema
                )
            print(f'{timer["pipeline"]:7.2f} seconds')

            display(self.table_extractions)

    def _mine_text_clicked(self, b):
        self.widgets['out'].clear_output()
        with self.widgets['out']:
            print('Mining request schema:')
            display(self.schema_request.schema)
            print("Running the mining pipeline...".ljust(50), end='', flush=True)
            text = self.widgets['input_text'].value
            self.table_extractions = self.textmining_pipeline(
                information=text,
                schema_df=self.schema_request.schema
            )
            display(self.table_extractions)

    def get_extracted_table(self):
        """Retrieve the table with the mining results.

        Returns
        -------
        results_table : pandas.DataFrame
            The table with the mining results.
        """
        if self.table_extractions is not None:
            results_table = self.table_extractions.copy()
        else:
            results_table = pd.DataFrame()

        return results_table

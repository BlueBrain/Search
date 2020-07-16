import io

from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import requests

from ..utils import Timer


class MiningWidget(widgets.VBox):

    def __init__(self, mining_server_url, article_saver=None, default_text=''):
        super().__init__()

        self.mining_server_url = mining_server_url
        self.article_saver = article_saver

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
            description='Mine This Text!',
            layout=widgets.Layout(width='auto'))
        self.widgets['mine_text'].on_click(self.mine_text_clicked)

        # "Mine Selected Articles" button
        self.widgets['mine_articles'] = widgets.Button(
            description='Mine Selected Articles!',
            layout=widgets.Layout(width='auto'))
        self.widgets['mine_articles'].on_click(self.mine_articles_clicked)

        # "Output Area" Widget
        self.widgets['out'] = widgets.Output(layout={'border': '0.5px solid black'})

    def _init_ui(self):
        self.children = [
            self.widgets['input_text'],
            widgets.HBox(children=[
                self.widgets['mine_text'],
                self.widgets['mine_articles']]),
            self.widgets['out'],
        ]

    def textmining_pipeline(self, information, debug=False):
        """Handle text mining server requests depending on the type of information.

        Parameters
        ----------
        information: str or list.
            Information can be either a raw string text, either a list of tuples
            (article_id, paragraph_id) related to the database.
        debug : bool
            If True, columns are not necessarily matching the specification. However, they
            contain debugging information. If False, then matching exactly the specification.

        Returns
        -------
        table_extractions: pd.DataFrame
            The final table. If `debug=True` then it contains all the metadata. If False then it
            only contains columns in the official specification.
        """
        if isinstance(information, list):
            response = requests.post(
                self.mining_server_url + '/database',
                json={"identifiers": information})
        elif isinstance(information, str):
            response = requests.post(
                self.mining_server_url + '/text',
                json={"text": information, "debug": debug})
        else:
            raise TypeError('Wrong type for the information!')

        table_extractions = None
        if response.headers["Content-Type"] == "text/csv":
            with io.StringIO(response.text) as f:
                table_extractions = pd.read_csv(f)
        else:
            print("Response content type is not text/csv.")
            print(response.headers)
            print(response.text)

        return table_extractions

    def mine_articles_clicked(self, b):
        self.widgets['out'].clear_output()

        if self.article_saver is None:
            with self.widgets['out']:
                print("No article saver was provided. Nothing to mine.")
            return

        with self.widgets['out']:
            timer = Timer(verbose=True)
            with timer('text retrieve'):
                self.article_saver.retrieve_text()
            with timer('casting to list'):
                chosen_text = self.article_saver.df_chosen_texts
                identifiers = [(row['article_id'], row['paragraph_id'])
                               for _, row in chosen_text.iterrows()]
                print(len(identifiers))
            with timer('server part'):
                self.table_extractions = self.textmining_pipeline(information=identifiers)
                display(self.table_extractions)

    def mine_text_clicked(self, b):
        self.widgets['out'].clear_output()
        with self.widgets['out']:
            text = self.widgets['input_text'].value
            self.table_extractions = self.textmining_pipeline(text)
            display(self.table_extractions)

    def get_extracted_table(self):
        if self.table_extractions is not None:
            return self.table_extractions.copy()
        else:
            return None

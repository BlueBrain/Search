"""Module for the mining widget."""
import io
import warnings

import ipywidgets as widgets
import pandas as pd
import requests
from IPython.display import HTML, display

from .._css import style
from ..utils import Timer


class MiningSchema:
    """The mining schema for the mining widget."""

    def __init__(self):
        self.columns = (
            "entity_type",
            "property",
            "property_type",
            "property_value_type",
            "ontology_source",
        )
        self.schema_df = pd.DataFrame(columns=self.columns)

    def add_entity(
        self,
        entity_type,
        property_name=None,
        property_type=None,
        property_value_type=None,
        ontology_source=None,
    ):
        """Add a new entity to the schema.

        A warning is issued for duplicate entities.

        Parameters
        ----------
        entity_type : str
            The entity type, for example "CHEMICAL".
        property_name: str, optional
            The property name, for example "isChiral".
        property_type : str, optional
            The property type, for example "ATTRIBUTE".
        property_value_type : str, optional
            The property value type, for example "BOOLEAN".
        ontology_source : str, optional
            The ontology source, for example "NCIT".
        """
        row = {
            "entity_type": entity_type,
            "property": property_name,
            "property_type": property_type,
            "property_value_type": property_value_type,
            "ontology_source": ontology_source,
        }
        # Make sure there are no duplicates to begin with
        self.schema_df.drop_duplicates(inplace=True, ignore_index=True)
        self.schema_df = self.schema_df.append(row, ignore_index=True)
        # If there are any duplicates at this point, then then it must have
        # come from the appended row.
        if any(self.schema_df.duplicated()):
            self.schema_df.drop_duplicates(inplace=True, ignore_index=True)
            warnings.warn("This entry already exists. No new entry was created.")

    def add_from_df(self, entity_df):
        """Add entities from a given dataframe.

        The data frame has to contain a column named "entity_type". Any
        columns matching the schema columns will be processed, all other
        columns will be ignored.

        Parameters
        ----------
        entity_df : pd.DataFrame
            The dataframe with new entities.
        """
        # The dataframe must contain the "entity_type" column
        if "entity_type" not in entity_df.columns:
            raise ValueError("Column named 'entity_type' not found.")

        # Collect all other valid columns
        valid_columns = []
        for column in self.schema_df:
            if column in entity_df.columns:
                valid_columns.append(column)
            else:
                warnings.warn(f"No column named {column} was found.")

        # Add new data to the schema
        for _, row in entity_df[valid_columns].iterrows():
            self.add_entity(
                row["entity_type"],
                property_name=row.get("property"),
                property_type=row.get("property_type"),
                property_value_type=row.get("property_value_type"),
                ontology_source=row.get("ontology_source"),
            )

    @property
    def df(self):
        """Get a dataframe with all entities.

        Returns
        -------
        schema_df : pd.DataFrame
            The dataframe with all entities.
        """
        return self.schema_df.copy()


class MiningWidget(widgets.VBox):
    """The mining widget.

    Parameters
    ----------
    mining_server_url : str
        The URL of the mining server.
    mining_schema : bbsearch.widgets.MiningSchema
        An object holding a dataframe with the requested mining df
        (entity, relation, attribute types).
    article_saver : bbsearch.widgets.ArticleSaver
        An instance of the article saver.
    default_text : string, optional
        The default text assign to the text area.
    use_cache : bool
        If True the mining server will use cached mining results stored in an
        SQL database. Should lead to major speedups.
    """

    def __init__(self, mining_server_url, mining_schema, article_saver=None, default_text='',
                 use_cache=True):
        super().__init__()

        self.mining_server_url = mining_server_url
        self.article_saver = article_saver
        self.mining_schema = mining_schema
        self.use_cache = use_cache

        # This is the output: csv table of extracted entities/relations.
        self.table_extractions = None

        # Define Widgets
        self.widgets = dict()

        self._init_widgets(default_text)
        self._init_ui()

        response = requests.post(
            self.mining_server_url + '/help',
        )
        self.database_name = response.json()['database'] if response.ok else None

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
            A dataframe with the requested mining df (entity, relation, attribute types).
        debug : bool
            If True, columns are not necessarily matching the specification. However, they
            contain debugging information. If False, then matching exactly the specification.

        Returns
        -------
        table_extractions: pd.DataFrame
            The final table. If `debug=True` then it contains all the metadata. If False then it
            only contains columns in the official specification.
        """
        schema_str = schema_df.to_csv(index=False)
        if isinstance(information, list):
            print(f"The widget is using database: {self.database_name}")
            response = requests.post(
                self.mining_server_url + '/database',
                json={"identifiers": information,
                      "schema": schema_str,
                      "use_cache": self.use_cache
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
            print('Mining request df:')
            display(self.mining_schema.df)
            print("Running the mining pipeline...".ljust(50), end='', flush=True)
            with timer("pipeline"):
                self.table_extractions = self.textmining_pipeline(
                    information=identifiers,
                    schema_df=self.mining_schema.df
                )
            print(f'{timer["pipeline"]:7.2f} seconds')

            display(self.table_extractions)

    def _mine_text_clicked(self, b):
        self.widgets['out'].clear_output()
        with self.widgets['out']:
            print('Mining request df:')
            display(self.mining_schema.df)
            print("Running the mining pipeline...".ljust(50), end='', flush=True)
            text = self.widgets['input_text'].value
            self.table_extractions = self.textmining_pipeline(
                information=text,
                schema_df=self.mining_schema.df
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

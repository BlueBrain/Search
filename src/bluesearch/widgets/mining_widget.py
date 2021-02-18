"""Module for the mining widget."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import io
import json
import pathlib

import ipywidgets as widgets
import pandas as pd
import requests
from IPython.display import HTML, display

from .._css import style
from ..utils import Timer

DEFAULT_MINING_TEXT = """Autophagy maintains tumour growth through circulating
 arginine. Autophagy captures intracellular components and delivers them to
 lysosomes, where they are degraded and recycled to sustain metabolism and to
 enable survival during starvation. Acute, whole-body deletion of the essential
 autophagy gene Atg7 in adult mice causes a systemic metabolic defect that
 manifests as starvation intolerance and gradual loss of white adipose tissue,
 liver glycogen and muscle mass. Cancer cells also benefit from autophagy.
 Deletion of essential autophagy genes impairs the metabolism, proliferation,
 survival and malignancy of spontaneous tumours in models of autochthonous
 cancer. Acute, systemic deletion of Atg7 or acute, systemic expression of a
 dominant-negative ATG4b in mice induces greater regression of KRAS-driven
 cancers than does tumour-specific autophagy deletion, which suggests that host
 autophagy promotes tumour growth.""".replace(
    "\n", ""
)


class MiningWidget(widgets.VBox):
    """The mining widget.

    Parameters
    ----------
    mining_server_url : str
        The URL of the mining server.
    mining_schema : bluesearch.widgets.MiningSchema
        The requested mining schema (entity, relation, attribute types).
    article_saver : bluesearch.widgets.ArticleSaver
        An instance of the article saver.
    default_text : string, optional
        The default text assign to the text area.
    use_cache : bool
        If True the mining server will use cached mining results stored in an
        SQL database. Should lead to major speedups.
    checkpoint_path : str or pathlib.Path, optional
        Path where checkpoints are saved to and loaded from. If `None`, defaults
        to `~/.cache/bluesearch/widgets_checkpoints` folder.
    """

    def __init__(
        self,
        mining_server_url,
        mining_schema,
        article_saver=None,
        default_text=DEFAULT_MINING_TEXT,
        use_cache=True,
        checkpoint_path=None,
    ):
        super().__init__()

        self.mining_server_url = mining_server_url
        self.article_saver = article_saver
        self.mining_schema = mining_schema
        self.use_cache = use_cache

        # This is the output: csv table of extracted entities/relations.
        self.table_extractions = None

        # Define Widgets
        self.widgets = {}

        self._init_widgets(default_text)
        self._init_ui()

        response = requests.post(
            self.mining_server_url + "/help",
        )
        if not response.ok:
            raise Exception(
                f"It seems there is an issue with the bbs mining server. Response "
                f"status is {response.status_code} : {response.text}"
            )

        response_json = response.json()

        self.database_name = response_json["database"]  # e.g "cord19_v47"
        self.mining_server_version = response_json["version"]  # e.g. "0.0.9.dev2+g69"

        if checkpoint_path is not None:
            self.checkpoint_path = pathlib.Path(checkpoint_path)
        else:
            self.checkpoint_path = (
                pathlib.Path.home() / ".cache" / "bluesearch" / "widgets_checkpoints"
            )
        self.checkpoint_path = self.checkpoint_path / "bbs_mining.json"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_widgets(self, default_text):
        # "Input Text" Widget
        self.widgets["input_text"] = widgets.Textarea(
            value=default_text, layout=widgets.Layout(width="75%", height="300px")
        )

        # "Mine This Text" button
        self.widgets["mine_text"] = widgets.Button(
            description="⚒️  Mine This Text!",
            layout=widgets.Layout(width="350px", height="50px"),
        )
        self.widgets["mine_text"].on_click(self._mine_text_clicked)
        self.widgets["mine_text"].add_class("bbs_button")

        # "Mine Selected Articles" button
        self.widgets["mine_articles"] = widgets.Button(
            description="⚒️  Mine Selected Articles!",
            layout=widgets.Layout(width="350px", height="50px"),
        )
        self.widgets["mine_articles"].on_click(self._mine_articles_clicked)
        self.widgets["mine_articles"].add_class("bbs_button")

        # Click to Save results
        self.widgets["save_button"] = widgets.Button(
            description="Save",
            icon="download",
            layout=widgets.Layout(width="172px", height="40px"),
        )
        self.widgets["save_button"].on_click(self._cb_bt_save)
        self.widgets["save_button"].add_class("bbs_button")

        # Click to Load results
        self.widgets["load_button"] = widgets.Button(
            description="Load",
            icon="upload",
            layout=widgets.Layout(width="172px", height="40px"),
        )
        self.widgets["load_button"].on_click(self._cb_bt_load)
        self.widgets["load_button"].add_class("bbs_button")

        # "Output Area" Widget
        self.widgets["out"] = widgets.Output(layout={"border": "0.5px solid black"})

        tabs = (
            (
                "Mine Articles",
                [
                    self.widgets["mine_articles"],
                ],
            ),
            (
                "Mine Text",
                [self.widgets["input_text"], self.widgets["mine_text"]],
            ),
        )

        tab_widget = widgets.Tab(children=[])
        for i, (tab_name, tab_children) in enumerate(tabs):
            tab_widget.children = tab_widget.children + (widgets.VBox(tab_children),)
            tab_widget.set_title(i, tab_name)
        self.widgets["mining"] = tab_widget

    def _init_ui(self):
        css_style = style.get_css_style()
        display(HTML(f"<style> {css_style} </style>"))

        self.children = [
            self.widgets["mining"],
            widgets.HBox(
                children=(self.widgets["save_button"], self.widgets["load_button"])
            ),
            self.widgets["out"],
        ]

    def textmining_pipeline(self, information, schema_df, debug=False):
        """Handle text mining server requests depending on the type of information.

        Parameters
        ----------
        information : str or list.
            Information can be either a raw string text, either a list of tuples
            (article_id, paragraph_id) related to the database.
        schema_df : pd.DataFrame
            A dataframe with the requested mining schema (entity, relation,
            attribute types).
        debug : bool
            If True, columns are not necessarily matching the specification.
            However, they contain debugging information. If False, then
            matching exactly the specification.

        Returns
        -------
        table_extractions : pd.DataFrame
            The final table. If `debug=True` then it contains all the
            metadata. If False then it only contains columns in the
            official specification.
        """
        schema_str = schema_df.to_csv(index=False)
        if isinstance(information, list):
            print(f"The widget is using database: {self.database_name}")
            response = requests.post(
                self.mining_server_url + "/database",
                json={
                    "identifiers": information,
                    "schema": schema_str,
                    "use_cache": self.use_cache,
                },
            )
        elif isinstance(information, str):
            response = requests.post(
                self.mining_server_url + "/text",
                json={"text": information, "schema": schema_str, "debug": debug},
            )
        else:
            raise TypeError("Wrong type for the information!")

        table_extractions = None
        if response.status_code == 200:
            response_dict = response.json()
            for warning_msg in response_dict["warnings"]:
                display(
                    HTML(
                        f'<div class="bbs_warning"> '
                        f"<b>WARNING!</b> {warning_msg} </div>"
                    )
                )
            with io.StringIO(response_dict["csv_extractions"]) as f:
                table_extractions = pd.read_csv(f)
        else:
            print("Server response is ERROR!")
            print(response.headers)
            print(response.text)

        return table_extractions

    def _mine_articles_clicked(self, b):
        self.widgets["out"].clear_output()

        if self.article_saver is None:
            with self.widgets["out"]:
                print("No article saver was provided. Nothing to mine.")
            return

        with self.widgets["out"]:
            timer = Timer()

            print("Collecting saved items...".ljust(50), end="", flush=True)
            with timer("collect items"):
                identifiers = self.article_saver.get_saved_items()

            print(f'{timer["collect items"]:7.2f} seconds')
            print("Mining request schema:")
            display(self.mining_schema.df)
            print("Running the mining pipeline...".ljust(50), end="", flush=True)
            with timer("pipeline"):
                self.table_extractions = self.textmining_pipeline(
                    information=identifiers, schema_df=self.mining_schema.df
                )
            print(f'{timer["pipeline"]:7.2f} seconds')

            display(self.table_extractions)

    def _mine_text_clicked(self, b):
        self.widgets["out"].clear_output()
        with self.widgets["out"]:
            print("Mining request schema:")
            display(self.mining_schema.df)
            print("Running the mining pipeline...".ljust(50), end="", flush=True)
            text = self.widgets["input_text"].value
            self.table_extractions = self.textmining_pipeline(
                information=text, schema_df=self.mining_schema.df
            )
            display(self.table_extractions)

    def _cb_bt_save(self, change_dict):
        with self.widgets["out"]:
            if self.table_extractions is None:
                message = """No mining results available. Did you forget
                             to run the mining pipeline on your selected
                             articles or text?"""
                display(HTML(f'<div class="bbs_error"> <b>ERROR!</b> {message} </div>'))
                return
            display(HTML("Saving mining results to disk...   "))
            data = {
                "mining_widget_extractions": self.table_extractions.to_dict(),
                "database_name": self.database_name,
                "mining_server_version": self.mining_server_version,
            }
            with self.checkpoint_path.open("w") as f:
                json.dump(data, f)
            display(HTML('<b class="bbs_success"> DONE!</b></br>'))

    def _cb_bt_load(self, change_dict):
        with self.widgets["out"]:
            self.widgets["out"].clear_output()
            if not self.checkpoint_path.exists():
                message = """No checkpoint file found to load. Did you forget to
                            save your mining results?"""
                display(
                    HTML(f'<div class="bbs_error"> ' f"<b>ERROR!</b> {message} </div>")
                )
                return
            display(HTML("Loading mining results from disk...   "))
            with self.checkpoint_path.open("r") as f:
                data = json.load(f)
            self.table_extractions = pd.DataFrame(data["mining_widget_extractions"])
            display(HTML('<b class="bbs_success"> DONE!</b></br>'))

            vers_load = data["mining_server_version"]
            vers_curr = self.mining_server_version
            db_load = data["database_name"]
            db_curr = self.database_name
            if db_load != db_curr or vers_load != vers_curr:
                message = f"""Loaded data from
                        <ul>
                            <li> mining server version = {vers_load} </li>
                            <li> database version = {db_load} </li>
                        </ul>
                        but current widget is connected to
                        <ul>
                            <li> mining server version = {vers_curr} </li>
                            <li> database version = {db_curr} </li>
                        </ul>
                        """
                display(
                    HTML(
                        f'<div class="bbs_warning"> '
                        f"<b>WARNING!</b> {message} </div>"
                    )
                )

            display(self.table_extractions)

    def _cb_chkb_show_mine_text_fct(self, change_dict):
        if change_dict["new"]:
            self.widgets["mine_text_fct"].layout.display = "block"
        else:
            self.widgets["mine_text_fct"].layout.display = "none"

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

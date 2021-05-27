"""The Search widget."""

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

import datetime
import enum
import functools
import json
import logging
import math
import pathlib
import sys
import textwrap
from urllib.parse import quote

import ipywidgets as widgets
import pandas as pd
import requests
from IPython.display import HTML, display

from .._css import style
from ..sql import (
    get_titles,
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
    bbs_search_url : str
        The URL of the bbs_search server.
    bbs_mysql_engine : sqlalchemy.engine.Engine
        Engine for connections to the bbs_mysql server.
    article_saver : bluesearch.widgets.ArticleSaver, optional
        If specified, this article saver will keep all the article_id
        of interest for the user during the different queries.
    results_per_page : int, optional
        The number of results to display per results page.
    checkpoint_path : str or pathlib.Path, optional
        Path where checkpoints are saved to and loaded from. If `None`, defaults
        to `~/.cache/bluesearch/widgets_checkpoints`.
    """

    def __init__(
        self,
        bbs_search_url,
        bbs_mysql_engine,
        article_saver=None,
        results_per_page=10,
        checkpoint_path=None,
    ):
        super().__init__()

        self.bbs_search_url = bbs_search_url
        self.bbs_mysql_engine = bbs_mysql_engine
        self.article_saver = article_saver
        self.results_per_page = max(1, results_per_page)
        self.n_pages = 1
        self.current_page = -1

        self.saving_labels = {
            _Save.NOTHING: "Do not take this article",
            _Save.PARAGRAPH: "Extract the paragraph",
            _Save.ARTICLE: "Extract the entire article",
        }

        self.radio_buttons = []
        self.current_sentence_ids = []
        self.history = []

        response = requests.post(
            self.bbs_search_url + "/help",
        )
        if not response.ok:
            raise Exception(
                f"It seems there is an issue with the bbs search server. Response "
                f"status is {response.status_code} : {response.text}"
            )

        response_json = response.json()

        self.supported_models = response_json["supported_models"]
        self.database_name = response_json["database"]  # e.g "cord19_v47"
        self.search_server_version = response_json["version"]  # e.g. "0.0.9.dev2+g69"

        self.widgets_style = {"description_width": "initial"}
        self.widgets = {}
        self._init_widgets()
        self._init_ui()

        if checkpoint_path is not None:
            self.checkpoint_path = pathlib.Path(checkpoint_path)
        else:
            self.checkpoint_path = (
                pathlib.Path.home() / ".cache" / "bluesearch" / "widgets_checkpoints"
            )
        self.checkpoint_path = self.checkpoint_path / "bbs_search.json"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_widgets(self):
        """Initialize widget dictionary."""
        # Select model to compute Sentence Embeddings
        self.widgets["sent_embedder"] = widgets.RadioButtons(
            options=self.supported_models,
            description="Model for Sentence Embedding",
            style=self.widgets_style,
            layout=widgets.Layout(width="450px", height="50px"),
        )

        # Select granularity of the search
        self.widgets["granularity"] = widgets.ToggleButtons(
            options=["sentences", "articles"],
            value="articles",
            disabled=False,
            style={"description_width": "initial", "button_width": "80px"},
            description="",
        )

        # Select n. of top results to return
        self.widgets["top_results"] = widgets.widgets.IntText(
            value=20, description="Show top ", style=self.widgets_style
        )

        # Choose whether to print whole paragraph containing sentence
        # highlighted, or just the sentence
        self.widgets["print_paragraph"] = widgets.Checkbox(
            value=True, description="Show whole paragraph", style=self.widgets_style
        )

        # Enter Query
        self.widgets["query_text"] = widgets.Textarea(
            value="Glucose is a risk factor for COVID-19",
            layout=widgets.Layout(width="90%", height="80px"),
            description="Query",
            style=self.widgets_style,
        )

        self.widgets["has_journal"] = widgets.Checkbox(
            description="Only articles from journals",
            value=True,
            style=self.widgets_style,
        )

        self.widgets["is_english"] = widgets.Checkbox(
            description="Only articles in English", value=True, style=self.widgets_style
        )

        self.widgets["discard_bad_sentences"] = widgets.Checkbox(
            description="Discard sentences flagged as bad quality",
            value=True,
            style=self.widgets_style,
        )

        self.widgets["date_range"] = widgets.IntRangeSlider(
            description="Date Range:",
            continuous_update=False,
            min=1850,
            max=2020,
            value=(2000, 2020),
            layout=widgets.Layout(width="80ch"),
            style=self.widgets_style,
        )
        # Enter Deprioritization Query
        self.widgets["deprioritize_text"] = widgets.Textarea(
            value="",
            layout=widgets.Layout(width="90%", height="80px"),
            description="Deprioritize",
            style=self.widgets_style,
        )

        # Select Deprioritization Strength
        self.widgets["deprioritize_strength"] = widgets.RadioButtons(
            options=[
                "None",
                "Mild",
                "Stronger",
            ],  # ['None', 'Weak', 'Mild', 'Strong', 'Stronger']
            disabled=False,
            style={"description_width": "initial", "button_width": "80px"},
            description="Deprioritization strength",
        )

        # Enter Substrings Exclusions
        self.widgets["exclusion_text"] = widgets.Textarea(
            layout=widgets.Layout(width="90%"),
            value="",
            style=self.widgets_style,
            description="Substring Exclusion (newline separated): ",
            rows=5,
        )
        self.widgets["exclusion_text"].layout.display = "none"

        self.widgets["inclusion_text"] = widgets.Textarea(
            layout=widgets.Layout(width="90%"),
            value="",
            style=self.widgets_style,
            description="Exact phrase matching:",
            rows=5,
            placeholder=textwrap.dedent(
                """
                    Case insensitive,  one phrase per line. Valid phrases are:
                    1. Single word                      : glucose
                    2. Multiple words                   : risk factor
                    3. Single word with variable suffix : molecul*
                       (matches "molecule", "molecules", "molecular")
                    """
            ).strip(),
        )

        self.widgets["default_value_article_saver"] = widgets.RadioButtons(
            options=[
                (self.saving_labels[_Save.NOTHING], _Save.NOTHING),
                (self.saving_labels[_Save.PARAGRAPH], _Save.PARAGRAPH),
                (self.saving_labels[_Save.ARTICLE], _Save.ARTICLE),
            ],
            value=_Save.ARTICLE,
            disabled=False,
            style={"description_width": "initial", "button_width": "200px"},
            description="Default saving: ",
        )

        # Click to run Information Retrieval!
        self.widgets["investigate_button"] = widgets.Button(
            description="📚 Search Literature!",
            layout=widgets.Layout(width="350px", height="50px"),
        )
        self.widgets["investigate_button"].add_class("bbs_button")

        # Click to Save results
        self.widgets["save_button"] = widgets.Button(
            description="Save",
            icon="download",
            layout=widgets.Layout(width="172px", height="40px"),
        )
        self.widgets["save_button"].add_class("bbs_button")

        # Click to Load results
        self.widgets["load_button"] = widgets.Button(
            description="Load",
            icon="upload",
            layout=widgets.Layout(width="172px", height="40px"),
        )
        self.widgets["load_button"].add_class("bbs_button")

        # Click to run Generate Report!
        self.widgets["report_button"] = widgets.Button(
            description="Generate Report of Search Results",
            layout=widgets.Layout(width="50%"),
        )

        self.widgets["articles_button"] = widgets.Button(
            description="Generate Report of Selected Articles",
            layout=widgets.Layout(width="50%"),
        )

        # Output Area
        self.widgets["out"] = widgets.Output(layout={"border": "1px solid black"})

        # Status Area
        self.widgets["status"] = widgets.Output(
            layout={"border": "1px solid black", "flex": "1"}
        )
        self.widgets["status_clear"] = widgets.Button(
            description="Clear", layout={"max_width": "100px"}
        )
        self.widgets["status_clear"].on_click(
            lambda b: self.widgets["status"].clear_output()
        )

        # Page buttons
        self.widgets["page_back"] = widgets.Button(
            description="←", layout={"width": "auto"}
        )
        self.widgets["page_label"] = widgets.Label(value="Page - of -")
        self.widgets["page_forward"] = widgets.Button(
            description="→", layout={"width": "auto"}
        )
        self.widgets["page_back"].on_click(
            lambda b: self.set_page(self.current_page - 1)
        )
        self.widgets["page_forward"].on_click(
            lambda b: self.set_page(self.current_page + 1)
        )

        # Put advanced settings to a tab
        tabs = (
            (
                "Search / View",
                [
                    self.widgets["sent_embedder"],
                    widgets.HBox(
                        children=[
                            self.widgets["top_results"],
                            self.widgets["granularity"],
                        ]
                    ),
                    self.widgets["print_paragraph"],
                    self.widgets["default_value_article_saver"],
                ],
            ),
            (
                "Filtering",
                [
                    self.widgets["has_journal"],
                    self.widgets["is_english"],
                    self.widgets["discard_bad_sentences"],
                    self.widgets["date_range"],
                    self.widgets["deprioritize_text"],
                    self.widgets["deprioritize_strength"],
                    self.widgets["exclusion_text"],
                    self.widgets["inclusion_text"],
                ],
            ),
        )
        tab_widget = widgets.Tab(children=[])
        tab_widget.layout.display = "none"
        for i, (tab_name, tab_children) in enumerate(tabs):
            tab_widget.children = tab_widget.children + (widgets.VBox(tab_children),)
            tab_widget.set_title(i, tab_name)
        self.widgets["advanced_settings"] = tab_widget

        # Disable advanced settings checkbox
        self.widgets["show_advanced_chb"] = widgets.Checkbox(
            value=False,
            description="Show advanced settings",
        )

        # Callbacks
        self.widgets["investigate_button"].on_click(self._cb_bt_investigate)
        self.widgets["save_button"].on_click(self._cb_bt_save)
        self.widgets["load_button"].on_click(self._cb_bt_load)
        self.widgets["report_button"].on_click(self._cb_bt_make_report_search)
        self.widgets["articles_button"].on_click(self._cb_bt_make_report_article_saver)
        self.widgets["show_advanced_chb"].observe(self._cb_chkb_advanced, names="value")

    def _init_ui(self):
        css_style = style.get_css_style()
        display(HTML(f"<style> {css_style} </style>"))

        page_selection = widgets.HBox(
            children=[
                self.widgets["page_back"],
                self.widgets["page_label"],
                self.widgets["page_forward"],
            ]
        )

        self.children = [
            self.widgets["query_text"],
            self.widgets["show_advanced_chb"],
            self.widgets["advanced_settings"],
            self.widgets["investigate_button"],
            widgets.HBox(
                children=(self.widgets["save_button"], self.widgets["load_button"])
            ),
            page_selection,
            self.widgets["out"],
            page_selection,
            widgets.HBox(
                children=(self.widgets["status"], self.widgets["status_clear"])
            ),
            self.widgets["report_button"],
            self.widgets["articles_button"],
        ]

        with self.widgets["out"]:
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
        sentence : str
            The sentence to highlight.

        Returns
        -------
        formatted_paragraph : str
            The paragraph containing `sentence` with the sentence highlighted
            in color
        """
        start = paragraph.index(sentence)
        end = start + len(sentence)
        highlighted_paragraph = f"""
            <div class="paragraph">
                {paragraph[:start]}
                <div class="paragraph_emph"> {paragraph[start:end]} </div>
                {paragraph[end:]}
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
        sentence = retrieve_sentences_from_sentence_ids(
            sentence_ids=(sentence_id,), engine=self.bbs_mysql_engine
        )
        article_id, section_name, text, paragraph_id = sentence.iloc[0][
            ["article_id", "section_name", "text", "paragraph_pos_in_article"]
        ]

        article = retrieve_article_metadata_from_article_id(
            article_id=article_id, engine=self.bbs_mysql_engine
        )
        article_auth, article_title, ref = article.iloc[0][["authors", "title", "url"]]

        try:
            article_auth = article_auth.split(";")[0] + " et al."
        except AttributeError:
            article_auth = ""

        ref = (
            ref.split(";")[0]
            if ref is not None
            else "https://www.google.com/search?q=" + quote(article_title)
        )
        section_name = section_name or ""

        result_info = {
            "sentence_id": sentence_id,
            "paragraph_id": int(paragraph_id),
            "article_id": article_id,
            "article_title": article_title,
            "article_auth": article_auth,
            "ref": ref,
            "section_name": section_name,
            "text": text,
        }

        return result_info

    def print_single_result(self, result_info, print_whole_paragraph):
        """Retrieve metadata and complete the report with HTML string given sentence_id.

        Parameters
        ----------
        result_info : dict
            The information for a single result obtained by calling
            `_fetch_result_info`.
        print_whole_paragraph : bool
            If true, the whole paragraph will be displayed in the results of the widget.

        Returns
        -------
        article_metadata : str
            Formatted string containing the metadata of the article.
        formatted_output : str
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
                paragraph = retrieve_paragraph_from_sentence_id(
                    sentence_id, self.bbs_mysql_engine
                )
                formatted_output = self.highlight_in_paragraph(paragraph, text)
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

    def _collect_search_configuration(self):
        """Read the search configuration from the widget components.

        Returns
        -------
        search_configuration : dict
            The search configuration.
        """
        search_configuration = {
            "which_model": self.widgets["sent_embedder"].value,
            "k": self.widgets["top_results"].value,
            "query_text": self.widgets["query_text"].value,
            "granularity": self.widgets["granularity"].value,
            "has_journal": self.widgets["has_journal"].value,
            "is_english": self.widgets["is_english"].value,
            "discard_bad_sentences": self.widgets["discard_bad_sentences"].value,
            "date_range": self.widgets["date_range"].value,
            "deprioritize_text": self.widgets["deprioritize_text"].value,
            "deprioritize_strength": self.widgets["deprioritize_strength"].value,
            "exclusion_text": self.widgets["exclusion_text"].value
            if "exclusion_text" in self.widgets.keys()
            else "",
            "inclusion_text": self.widgets["inclusion_text"].value,
            "verbose": False,
        }

        return search_configuration

    def _query_search_server(self, search_configuration):
        """Query the search server.

        Parameters
        ----------
        search_configuration : dict
            The search configuration.

        Returns
        -------
        result : dict or None
            If the query was successful then a dictionary with the query
            results is returned. In case of an error None is returned.
        """
        try:
            response = requests.post(self.bbs_search_url, json=search_configuration)
            response.raise_for_status()  # if not response.ok
        except requests.ConnectionError as e:
            print(f"Could not connect to the search server.\n\n{e}", file=sys.stderr)
            result = None
        except requests.HTTPError as e:
            # raised by response.raise_for_status()
            print(f"There was an HTTP error.\n\n{e}", file=sys.stderr)
            result = None
        else:
            result = response.json()

        return result

    def _cb_bt_investigate(self, change_dict):
        """Investigate button callback."""
        # Clear output and show waiting message
        timer = Timer()
        self.widgets["out"].clear_output()
        self.widgets["status"].clear_output()
        with self.widgets["status"]:
            search_configuration = self._collect_search_configuration()
            header = f'Query: "{search_configuration["query_text"]}"'
            print(header)
            print("-" * len(header))

            print(f"INFO: Database {self.database_name} is used for the search query.")
            print("Sending query to server...".ljust(50), end="", flush=True)
            with timer("server query"):
                response = self._query_search_server(search_configuration)
                if response is None:
                    return
                else:
                    self.current_sentence_ids = response["sentence_ids"]
            print(f'{timer["server query"]:7.2f} seconds')

            print("Processing search results...".ljust(50), end="", flush=True)
            with timer("default saving"):
                self._process_search_results()
            print(f'{timer["default saving"]:7.2f} seconds')

            print("Updating the results display...".ljust(50), end="", flush=True)
            with timer("update page"):
                self.n_pages = math.ceil(
                    len(self.current_sentence_ids) / self.results_per_page
                )
                self.set_page(0, force=True)
            print(f'{timer["update page"]:7.2f} seconds')

            print("Done.")

    def _cb_bt_save(self, change_dict):
        with self.widgets["status"]:
            self.widgets["status"].clear_output()
            if not self.article_saver.state or not self.history:
                message = """No articles or paragraphs selected. Did you forget
                             to run your query or select some search results?"""
                display(HTML(f'<div class="bbs_error"> <b>ERROR!</b> {message} </div>'))
                return
            display(HTML("Saving search results to disk...   "))
            data = {
                "article_saver_state": list(self.article_saver.state),
                "search_widget_history": self.history,
                "database_name": self.database_name,
                "search_server_version": self.search_server_version,
            }
            with self.checkpoint_path.open("w") as f:
                json.dump(data, f)
            self.widgets["status"].clear_output()
            display(
                HTML(
                    "Saving search results to disk... "
                    '<b class="bbs_success"> DONE!</b></br>'
                )
            )

    def _cb_bt_load(self, change_dict):
        with self.widgets["status"]:
            self.widgets["status"].clear_output()
            if not self.checkpoint_path.exists():
                message = """No checkpoint file found to load. Did you forget to
                            save your search results?"""
                display(
                    HTML(f'<div class="bbs_error"> ' f"<b>ERROR!</b> {message} </div>")
                )
                return
            display(HTML("Loading search results from disk...   "))
            with self.checkpoint_path.open("r") as f:
                data = json.load(f)
            self.article_saver.state = {tuple(t) for t in data["article_saver_state"]}
            self.history = data["search_widget_history"]
            self.widgets["status"].clear_output()
            display(
                HTML(
                    "Loading search results from disk...   "
                    '<b class="bbs_success"> DONE!</b></br>'
                )
            )

            vers_load = data["search_server_version"]
            vers_curr = self.search_server_version
            db_load = data["database_name"]
            db_curr = self.database_name
            if db_load != db_curr or vers_load != vers_curr:
                message = f"""Loaded data from
                        <ul>
                            <li> search server version = {vers_load} </li>
                            <li> database version = {db_load} </li>
                        </ul>
                        but current widget is connected to
                        <ul>
                            <li> search server version = {vers_curr} </li>
                            <li> database version = {db_curr} </li>
                        </ul>
                        """
                display(
                    HTML(
                        f'<div class="bbs_warning"> '
                        f"<b>WARNING!</b> {message} </div>"
                    )
                )

    def _process_search_results(self):
        """Flag items corresponding to sentence IDs for saving.

        The default saving strategy is given by the corresponding
        saving setting widget state.

        This also updates the search history.
        """
        default_saving_value = self.widgets["default_value_article_saver"].value
        sentence_df = retrieve_sentences_from_sentence_ids(
            sentence_ids=self.current_sentence_ids,
            engine=self.bbs_mysql_engine,
            keep_order=True,
        )

        for row in sentence_df.itertuples(index=False):
            self.history.append(
                (row.article_id, row.paragraph_pos_in_article, row.sentence_id)
            )
            if self.article_saver is not None:
                if default_saving_value == _Save.ARTICLE:
                    self.article_saver.add_article(row.article_id)
                elif default_saving_value == _Save.PARAGRAPH:
                    self.article_saver.add_paragraph(
                        row.article_id, row.paragraph_pos_in_article
                    )

    def saved_results(self):
        """Get all search results that were flagged for saving.

        Returns
        -------
        saved_items_df : pd.DataFrame
            A data frame with all saved search results.
        """
        # Get all titles first
        article_ids = [article_id for article_id, *_ in self.history]
        titles = get_titles(article_ids, self.bbs_mysql_engine)

        # For each item in history get its saving status
        rows = []
        columns = ["Article ID", "Paragraph #", "Paragraph", "Article", "Title"]
        markers = {True: "✓", False: ""}
        for article_id, paragraph_pos, _sentence_id in self.history:
            # Get saving status from the article saver
            if self.article_saver is None:
                paragraph_saved = False
                article_saved = False
            else:
                paragraph_saved = self.article_saver.has_paragraph(
                    article_id, paragraph_pos
                )
                article_saved = self.article_saver.has_article(article_id)

            # Dont' show paragraph position if no paragraph saved
            if not paragraph_saved:
                paragraph_pos = ""

            # Don't show items that are not saved
            if any([paragraph_saved, article_saved]):
                row = (
                    article_id,
                    paragraph_pos,
                    markers[paragraph_saved],
                    markers[article_saved],
                    titles[article_id],
                )
                rows.append(row)

        saved_items_df = pd.DataFrame(rows, columns=columns)

        return saved_items_df

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
            page_label = f"Page {self.current_page + 1} of {self.n_pages}"
            self.widgets["page_label"].value = page_label
            self._update_page_display()

    def _update_page_display(self):
        with self.widgets["out"]:
            print_whole_paragraph = self.widgets["print_paragraph"].value
            self.radio_buttons = []

            self.widgets["out"].clear_output()
            start = self.current_page * self.results_per_page
            end = start + self.results_per_page
            for sentence_id in self.current_sentence_ids[start:end]:
                result_info = self._fetch_result_info(sentence_id)
                article_metadata, formatted_output = self.print_single_result(
                    result_info, print_whole_paragraph
                )
                if self.article_saver:
                    chk_article, chk_paragraph = self._create_saving_checkboxes(
                        result_info["article_id"], result_info["paragraph_id"]
                    )

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

    def _cb_chkb_advanced(self, change_dict):
        if change_dict["new"]:
            self.widgets["advanced_settings"].layout.display = "block"
        else:
            self.widgets["advanced_settings"].layout.display = "none"

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
                paragraph_id=paragraph_id,
            ),
            names="value",
        )
        chk_article.observe(
            handler=functools.partial(
                self._cb_chkb_save_article, article_id=article_id
            ),
            names="value",
        )

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

    def _cb_bt_make_report_article_saver(self, change_dict):
        """Create the saved articles report."""
        with self.widgets["status"]:
            print()
            print("Creating the saved results report... ")
            out_file = self.article_saver.make_report()
            print(f"Done. Report saved to {out_file}.")

    def _cb_bt_make_report_search(self, change_dict):
        """Create the report of the search."""
        with self.widgets["status"]:
            print()
            print("Creating the search results report... ")

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

            print_whole_paragraph = self.widgets["print_paragraph"].value
            report = ""
            for sentence_id in self.current_sentence_ids:
                result_info = self._fetch_result_info(sentence_id)
                article_metadata, formatted_output = self.print_single_result(
                    result_info, print_whole_paragraph
                )
                report += article_metadata + formatted_output + "<br>"

            results_section = f"<h1> Results </h1> {report}"

            css_style = style.get_css_style()

            output_file = pathlib.Path(f"report_{datetime.datetime.now()}.html")
            with output_file.open("w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write(f"<style> {css_style} </style>")
                f.write(hyperparameters_section)
                f.write(results_section)
            print(f"Done. Report saved to {output_file}.")

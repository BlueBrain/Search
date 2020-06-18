from collections import OrderedDict
import io
import warnings

from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import requests


class MiningWidget(widgets.VBox):

    def __init__(self, text_mining_url, article_saver, default_text):
        super().__init__()

        self.text_mining_url = text_mining_url
        self.article_saver = article_saver
        self.default_text = default_text
        self.table_extractions = None
        self.widgets = OrderedDict()

        self._init_ui()

    def mining_pipeline(self, text, article_id=None, return_prob=False, debug=False):
        request_json = {
            "text": text,
            "article_id": article_id,
            "return_prob": return_prob,
            "debug": debug,
        }
        response = requests.post(self.text_mining_url, json=request_json)
        if response.headers["Content-Type"] == "text/csv":
            with io.StringIO(response.text) as f:
                table_extractions = pd.read_csv(f)
        else:
            warnings.warn("Response content type is not text/csv.")
            table_extractions = None

        return table_extractions

    def _init_ui(self):
        self.widgets['articles_button'] = widgets.Button(
            description='Mine Selected Articles!',
            layout=widgets.Layout(width='60%'))
        self.widgets['input_text'] = widgets.Textarea(
            value=self.default_text,
            layout=widgets.Layout(width='75%', height='300px'))
        self.widgets['submit_button'] = widgets.Button(
            description='Mine This Text!',
            layout=widgets.Layout(width='30%'))
        self.widgets['out'] = widgets.Output(
            layout={'border': '0.5px solid black'})

        self.widgets['articles_button'].on_click(self._article_button_on_click)
        self.widgets['submit_button'].on_click(self._text_button_on_click)

        self.children = list(self.widgets.values())

    def _article_button_on_click(self, b):
        self.article_saver.retrieve_text()
        self.table_extractions = pd.DataFrame()
        columns = self.article_saver.df_chosen_texts[
            ['article_id', 'section_name', 'paragraph_id', 'text']]
        for article_id, section_name, paragraph_id, text in columns.values:
            text_identifier = f'{article_id}:"{section_name}":{paragraph_id}'
            new_extractions = self.mining_pipeline(
                text,
                article_id=text_identifier,
                return_prob=False)
            self.table_extractions = self.table_extractions.append(
                new_extractions,
                ignore_index=True)

        self.widgets['out'].clear_output()
        with self.widgets['out']:
            display(self.table_extractions)

    def _text_button_on_click(self, b):
        text = self.widgets['input_text'].value
        self.table_extractions = self.mining_pipeline(
            text,
            return_prob=False)

        self.widgets['out'].clear_output()
        with self.widgets['out']:
            display(self.table_extractions)


class TypeSelectionBox(widgets.VBox):

    def __init__(self, option_dict, label=None, **vbox_kwargs):
        super().__init__(**vbox_kwargs)
        self.checkboxes = OrderedDict()
        for type_name, (type_label, initial_value) in option_dict.items():
            checkbox = widgets.Checkbox(
                value=initial_value,
                description=type_label)
            self.checkboxes[type_name] = checkbox
        self.children = tuple(self.checkboxes.values())

        if label is not None:
            self.children = (widgets.Label(label),) + self.children

    def get_selected(self):
        selected = set()
        for name, checkbox in self.checkboxes.items():
            if checkbox.value:
                selected.add(name)
        return selected


class MiningConfigurationWidget(widgets.VBox):

    def __init__(self, available_entity_types, available_relation_types, **vbox_kwargs):
        super().__init__(**vbox_kwargs)
        selector_layout = {"border": "black dotted 1pt"}
        self.entity_selector = TypeSelectionBox(
            available_entity_types,
            label="Entity Types",
            layout=selector_layout)
        self.relation_selector = TypeSelectionBox(
            available_relation_types,
            label="Relation Types",
            layout=selector_layout)
        self.children = (
            widgets.Label("Select entity and relation types"),
            widgets.HBox(children=(self.entity_selector, self.relation_selector))
        )

    def get_selected_entity_types(self):
        return self.entity_selector.get_selected()

    def get_selected_relation_types(self):
        return self.relation_selector.get_selected()
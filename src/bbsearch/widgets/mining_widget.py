from collections import OrderedDict
from functools import partial
import io
import warnings

from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import requests


entity_type_d = {
    "CHEMICAL": "Chemical",
    "DRUG": "Drug",
    "PERSON": "Person",
    "COUNTRY": "Country",
}

relation_type_d = {
    "UPREGULATES": "upregulates",
    "DOWNREGULATES": "downregulates",
    "ACTIVATES": "activates",
    "INHIBITS": "inhibits",
    "IS_ANTAGONIST_OF": "is antagonist of",
    "LOVES": "loves",
    "HATES": "hates",
    "BORN_IN": "born in",
    "LIVES_IN": "lives in",
}

available_relation_types = {
    ("CHEMICAL", "DRUG"): {"UPREGULATES", "DOWNREGULATES", "ACTIVATES", "INHIBITS", "IS_ANTAGONIST_OF"},
    ("PERSON", "PERSON"): {"LOVES", "HATES"},
    ("PERSON", "COUNTRY"): {"BORN_IN", "LIVES_IN"}
}


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

    def __init__(self, options, label=None, **vbox_kwargs):
        super().__init__(**vbox_kwargs)
        self.options = options
        self.checkbox_listeners = []

        button_layout = widgets.Layout(width="30px", margin="0 0 0 5px")

        self.select_all = widgets.Button(description="", layout=button_layout, icon="check")
        self.deselect_all = widgets.Button(description="", layout=button_layout, icon="close")
        self.select_all.on_click(partial(self._change_all, new_state=True))
        self.deselect_all.on_click(partial(self._change_all, new_state=False))
        self.header_panel = widgets.HBox(children=(self.select_all, self.deselect_all))
        if label is not None:
            self.header_panel.children = self.header_panel.children + (widgets.Label(label),)

        self.checkboxes = OrderedDict()
        for type_name, type_label in options:
            checkbox = widgets.Checkbox(
                value=False,
                description=type_label,
                indent=False)
            self.checkboxes[type_name] = checkbox
            checkbox.observe(self._on_checkbox_change, names="value")

        self.button_box = widgets.HBox(children=(self.select_all, self.deselect_all))
        self.out = widgets.Output()

        self.children = (self.header_panel,) + tuple(self.checkboxes.values()) + (self.out,)

    def _on_checkbox_change(self, event):
        all_checkbox_states = {}
        for type_name, type_label in self.options:
            all_checkbox_states[type_name] = self.checkboxes[type_name].value

        for listener_fn in self.checkbox_listeners:
            listener_fn(all_checkbox_states)

    def _change_all(self, b, new_state):
        with self.out:
            for checkbox in self.checkboxes.values():
                checkbox.value = new_state

    def register_checkbox_listener(self, listener_fn):
        self.checkbox_listeners.append(listener_fn)

    def get_selected(self):
        selected = set()
        for name, checkbox in self.checkboxes.items():
            if checkbox.value:
                selected.add(name)
        return selected

    def set_enabled(self, enabled=True):
        for checkbox in self.checkboxes.values():
            checkbox.disabled = not enabled
        self.select_all.disabled = not enabled
        self.deselect_all.disabled = not enabled


class MiningConfigurationWidget(widgets.HBox):

    def __init__(self, entities_we_want, relations_we_want):
        super().__init__()

        self.right_panel = widgets.VBox()
        self.entity_selector = TypeSelectionBox(
            [(entity_type, entity_type_d[entity_type]) for entity_type in entities_we_want],
            label="All Entity Types",
            style={"border": "black dotted 1pt"})
        self.entity_selector.register_checkbox_listener(self.entity_change_callback)
        self.children = (
            self.entity_selector,
            self.right_panel
        )
        self.out = widgets.Output()

        self.relation_lists = OrderedDict()
        for entity_pair, relations in relations_we_want.items():
            label = "{} - {}".format(entity_type_d[entity_pair[0]], entity_type_d[entity_pair[1]])
            values = [(relation, relation_type_d[relation]) for relation in relations]
            self.relation_lists[entity_pair] = TypeSelectionBox(
                values,
                label=label)
        self.right_panel.children = (
            *self.relation_lists.values(),
            self.out,
        )

    def entity_change_callback(self, entity_statuses):
        with self.out:
            checked_entities = {entity for entity, status in entity_statuses.items() if status}
            for (entity_1, entity_2), selection_box in self.relation_lists.items():
                if entity_1 in checked_entities and entity_2 in checked_entities:
                    selection_box.set_enabled(True)
                else:
                    selection_box.set_enabled(False)

    def get_selected_entity_types(self):
        return self.entity_selector.get_selected()

    def get_selected_relation_types(self):
        selected = {}
        for entity_pair, box in self.relation_lists.items():
            selected[entity_pair] = box.get_selected()
        return selected

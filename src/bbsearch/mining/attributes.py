import collections
import json
import logging
import textwrap
import warnings

import pandas as pd
import requests
from IPython.display import HTML

from .entity import find_entities


logger = logging.getLogger(__name__)


class AttributeExtractor:

    def __init__(self, core_nlp_url, grobid_quantities_url, ee_model):
        logger.debug(f"{self.__class__.__name__} init")
        logger.debug(f"CoreNLP URL: {core_nlp_url}")
        logger.debug(f"Grobid Quantities URL: {grobid_quantities_url}")
        logger.debug(f"Entity Extraction Model:\n{ee_model.meta}")

        self.core_nlp_url = core_nlp_url
        self.grobid_quantities_url = grobid_quantities_url
        self.ee_model = ee_model

    @staticmethod
    def get_quantity_type(quantity):
        try:
            quantity_type = quantity['rawUnit']['type']
        except KeyError:
            try:
                quantity_type = quantity['normalizedUnit']['type']
            except KeyError:
                return ''

        return quantity_type

    def get_measurement_type(self, measurement):
        logger.debug("get_measurement_type")
        logger.debug(f"measurement:\n{measurement}")

        quantity_types = [self.get_quantity_type(quantity)
                          for quantity in self.iter_quantities(measurement)]
        logger.debug(f"quantity_types: {quantity_types}")
        if not len(set(quantity_types)) == 1:
            msg = f"""
            Measurement contains multiple quantity types.
            measurement:
            {measurement}
            quantity types found: {quantity_types}
            """
            logger.warning(textwrap.dedent(msg).strip())
        return quantity_types[0]

    def get_all_measurement_types(self, measurements):
        all_types = [self.get_measurement_type(m) for m in measurements]
        all_type_counts = collections.Counter(all_types)
        return all_type_counts

    def get_grobid_measurements(self, text):
        response = requests.post(
            self.grobid_quantities_url,
            files={'text': text})
        measurements = []

        if response.status_code != 200:
            msg = f"GROBID request problem. Code: {response.status_code}"
            warnings.warn(msg)
        else:
            response_json = json.loads(response.text)
            if 'measurements' in response_json:
                measurements = response_json['measurements']

        return measurements

    def annotate_quantities(self, text, measurements, width):
        css_styles = f"""
        <style>
        .number  {{
            display: inline-block;
            background: lightgreen;
            padding: 0.2em 0.5em;
            border-radius: 7px;
        }}
        .unit {{
            display: inline-block;
            background: pink;
            padding: 0.2em 0.5em;
            border-radius: 7px;
        }}
        .quantityType {{
            display: inline-block;
            background: yellow;
            font-variant:small-caps;
            padding: 0.2em 0.5em;
            border-radius: 7px;
        }}
        .fixedWidth {{
            width: {width}ch;
            text-align: justify;
        }}
        </style>
        """

        def annotate_quantity(quantity):
            annotations = []
            start = quantity['offsetStart']
            end = quantity['offsetEnd']
            formatted_text = f"<span class=\"number\">{text[start:end]}</span>"
            quantity_type = self.get_quantity_type(quantity)
            if quantity_type:
                formatted_text += f"<span class=\"quantityType\">[{quantity_type}]</span>"
            annotations.append([start, end, formatted_text])

            if 'rawUnit' in quantity:
                start = quantity['rawUnit']['offsetStart']
                end = quantity['rawUnit']['offsetEnd']
                annotations.append([start, end, f"<span class=\"unit\">{text[start:end]}</span>"])

            return annotations

        annotations = []
        for measurement in measurements:
            for quantity in self.iter_quantities(measurement):
                annotations += annotate_quantity(quantity)

        sorted(annotations, key=lambda x: x[0])
        annotated_text = ''
        last_idx = 0
        for start, end, quantity in annotations:
            annotated_text += text[last_idx:start] + quantity
            last_idx = end
        annotated_text += text[last_idx:]
        html = css_styles + f"<div class=\"fixedWidth\">" + annotated_text + "</div>"

        return HTML(html)

    @staticmethod
    def get_overlapping_token_ids(start, end, tokens):
        ids = []
        for token in tokens:
            start_inside = start <= token['characterOffsetBegin'] < end
            end_inside = start < token['characterOffsetEnd'] <= end
            if start_inside or end_inside:
                ids.append(token['index'])

        return ids

    @staticmethod
    def iter_quantities(measurement):
        if 'quantity' in measurement:
            yield measurement['quantity']
        elif 'quantities' in measurement:
            yield from measurement['quantities']
        elif 'quantityMost' in measurement or 'quantityLeast' in measurement:
            if 'quantityLeast' in measurement:
                yield measurement['quantityLeast']
            if 'quantityMost' in measurement:
                yield measurement['quantityMost']
        elif 'quantityBase' in measurement or 'quantityRange' in measurement:
            if 'quantityBase' in measurement:
                yield measurement['quantityBase']
            if 'quantityRange' in measurement:
                yield measurement['quantityRange']
        else:
            warnings.warn("no quantity in measurement")
            return

    def get_quantity_tokens(self, quantity, tokens):
        value_start = quantity["offsetStart"]
        value_end = quantity["offsetEnd"]
        ids = self.get_overlapping_token_ids(value_start, value_end, tokens)
        if "rawUnit" in quantity:
            unit_start = quantity["rawUnit"]["offsetStart"]
            unit_end = quantity["rawUnit"]["offsetEnd"]
            ids += self.get_overlapping_token_ids(unit_start, unit_end, tokens)

        return ids

    def get_measurement_tokens(self, measurement, tokens):
        ids = []

        for quantity in self.iter_quantities(measurement):
            ids += self.get_quantity_tokens(quantity, tokens)

        return ids

    def get_entity_tokens(self, entity, tokens):
        return self.get_overlapping_token_ids(
            entity.start_char,
            entity.end_char,
            tokens)

    @staticmethod
    def find_compound_parents(dependencies, tokens_d, token_idx):
        parents = []
        for link in dependencies:
            if link['dependent'] == token_idx and link['dep'] == "compound":
                parents.append(link['governor'])

        return parents

    @staticmethod
    def iter_parents(dependencies, token_idx):
        """
        Seems each node has at most one parent (verify!)

        It seems parent=0 means no parent
        """

        for link in dependencies:
            if link['dependent'] == token_idx:
                parent = link['governor']
                if parent != 0:
                    yield link['governor']

    def find_nn_parents(self, dependencies, tokens_d, token_idx):
        """
        Ascent the dependency tree until find a parent
        of type "NN". Do this for all parents. If, as
        it seems, each node has at most one parent, then
        the results will be either one index or no indices.
        """

        def get_nn(idx):
            if tokens_d[idx]['pos'] == 'NN':
                return [idx]
            else:
                nn_parents = []
                for new_idx in self.iter_parents(dependencies, idx):
                    nn_parents += get_nn(new_idx)
                return nn_parents

        results = []
        for parent_idx in self.iter_parents(dependencies, token_idx):
            results += get_nn(parent_idx)

        return results

    def find_all_parents(self, dependencies, tokens_d, tokens, parent_fn=None):
        if parent_fn is None:
            parent_fn = self.find_nn_parents

        parent_ids = []

        for token_idx in tokens:
            parent_ids += parent_fn(dependencies, tokens_d, token_idx)

        return parent_ids

    def extract_attributes(self, text):
        # NER
        doc = find_entities(text, self.ee_model)
        sent = list(doc.sents)[0]
        detected_entities = [ent for ent in sent.ents]
        logging.info("{} entities detected: {}".format(len(detected_entities), detected_entities))

        # Grobid Quantities
        measurements = self.get_grobid_measurements(text)
        logging.info("{} measurements detected".format(len(measurements)))

        # CoreNLP
        logging.info("Sending CoreNLP query...")
        try:
            response = requests.post(self.core_nlp_url, data=text)
            assert response.status_code == 200
            response_json = json.loads(response.text)
        except:
            response_json = {'sentences': []}
        logging.info("CoreNLP found {} sentences".format(len(response_json['sentences'])))

        # Analysis
        columns = ["entity", "entity_type", "attribute"]
        rows = []

        for entity in detected_entities:
            for measurement in measurements:
                for corenlp_sentence in response_json['sentences']:
                    tokens = corenlp_sentence['tokens']
                    dependencies = corenlp_sentence['basicDependencies']
                    tokens_d = {token['index']: token for token in tokens}

                    measurement_ids = self.get_measurement_tokens(measurement, tokens)
                    ne_ids = self.get_entity_tokens(entity, tokens)

                    measurement_parents = self.find_all_parents(dependencies, tokens_d, measurement_ids)
                    ne_parents = self.find_all_parents(dependencies, tokens_d, ne_ids)

                    if len(set(measurement_parents) & set(ne_parents)) > 0:
                        row = {
                            "entity": entity.text,
                            "entity_type": entity.label_,
                            "attribute": measurement
                        }
                        rows.append(row)

        df_attributes = pd.DataFrame(rows, columns=columns)

        return df_attributes

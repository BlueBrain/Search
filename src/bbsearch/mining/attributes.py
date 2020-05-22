"""Classes and functions for Attribute Extraction.

The main workhorse is the class `AttributeExtraction`.
"""

import collections
import json
import logging
import warnings

import pandas as pd
import requests
from IPython.display import HTML


logger = logging.getLogger(__name__)


class AttributeExtractor:
    """Extract and analyze attributes in a given text."""

    def __init__(self, core_nlp_url, grobid_quantities_url, ee_model):
        """Initialize the class.

        Parameters
        ----------
        core_nlp_url : str
            The URL of the CoreNLP server.
        grobid_quantities_url : str
            The URL of the Grobid Quantities server.
        ee_model : spacy.language.Language
            The spacy model for name entity extraction
        """
        logger.debug(f"{self.__class__.__name__} init")
        logger.debug(f"CoreNLP URL: {core_nlp_url}")
        logger.debug(f"Grobid Quantities URL: {grobid_quantities_url}")
        logger.debug(f"Entity Extraction Model:\n{ee_model.meta}")

        self.core_nlp_url = core_nlp_url
        self.grobid_quantities_url = grobid_quantities_url
        self.ee_model = ee_model

    @staticmethod
    def get_quantity_type(quantity):
        """Get the type of a Grobid quantity.

        The top-level Grobid object is a measurement. A measurement can
        contain one ore more than one quantities.

        Some Grobid quantities have a type attached to them, e.g.
        "mass", "concentration", etc. This is the type that is
        returned. For quantities without a type an empty string
        is returned.

        Parameters
        ----------
        quantity : dict
            A Grobid quantity.

        Returns
        -------
        quantity_type : str
            The type of the quantity.
        """
        try:
            quantity_type = quantity['rawUnit']['type']
        except KeyError:
            try:
                quantity_type = quantity['normalizedUnit']['type']
            except KeyError:
                quantity_type = ''

        return quantity_type

    def get_measurement_type(self, measurement):
        """Get the type of a Grobid measurement.

        For measurements with multiple quantities the
        most common type is returned. In case of ties
        the empty type always loses.

        Parameters
        ----------
        measurement : dict
            A Grobid measurement.

        Returns
        -------
        measurement_type : str
            The type of the Grobid measurement.
        """
        logger.debug("get_measurement_type")
        logger.debug(f"measurement:\n{measurement}")

        quantity_types = [self.get_quantity_type(quantity)
                          for quantity in self.iter_quantities(measurement)]
        logger.debug(f"quantity_types: {quantity_types}")

        quantity_type_counts = collections.Counter(quantity_types)
        most_common_quantity_types = sorted(
            quantity_type_counts.most_common(),
            key=lambda t_cnt: (-t_cnt[1], int(t_cnt[0] == '')))
        measurement_type = most_common_quantity_types[0][0]

        return measurement_type

    def count_measurement_types(self, measurements):
        """Count types of all given measurements.

        Parameters
        ----------
        measurements : list
            A list of Grobid measurements.

        Returns
        -------
        all_type_counts : collections.Counter
            The counts of al
        """
        all_types = [self.get_measurement_type(m) for m in measurements]
        all_type_counts = collections.Counter(all_types)
        return all_type_counts

    def get_grobid_measurements(self, text):
        """Get measurements for text form Grobid server.

        Parameters
        ----------
        text : str
            The text for the query.

        Returns
        -------
        measurements : list_like
            All Grobid measurements extracted from the given text.
        """
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
        """Annotate measurements in text using HTML/CSS styles.

        Parameters
        ----------
        text : str
            The text to annotate.
        measurements : list
            The Grobid measurements for the text. It is assumed
            that these measurements were obtained by calling
            `get_grobid_measurements(text)`.
        width : int
            The width of the output <div> in characters.

        Returns
        -------
        output : IPython.core.display.HTML
            The annotated text.
        """
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
        html = css_styles + "<div class=\"fixedWidth\">" + annotated_text + "</div>"

        output = HTML(html)

        return output

    @staticmethod
    def get_overlapping_token_ids(start, end, tokens):
        """Find tokens intersecting the interval [start, end).

        CoreNLP breaks a given text down into sentences, and
        each sentence is broken down into tokens. These can
        be accessed by `response['sentences'][sentence_id]['tokens'].

        Each token corresponds to a position in the original text.
        This method determines which tokens would intersect a
        a given slice of this text.

        Parameters
        ----------
        start : int
            The left boundary of the interval.
        end : int
            The right boundary of the interval.
        tokens : list
            The CoreNLP sentence tokens.

        Returns
        -------
        ids : list
            A list of token indices that overlap with the
            given interval.
        """
        ids = []
        for token in tokens:
            start_inside = start <= token['characterOffsetBegin'] < end
            end_inside = start < token['characterOffsetEnd'] <= end
            if start_inside or end_inside:
                ids.append(token['index'])

        return ids

    @staticmethod
    def iter_quantities(measurement):
        """Iterate over quantities in a Grobid measurement.

        Parameters
        ----------
        measurement : dict
            A Grobid measurement.

        Yields
        ------
        quantity : dict
            A Grobid quantity in the given measurement.

        """
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
        """Associate a Grobid quantity to CoreNLP tokens.

        Both the quantity and the tokens should originate
        from exactly the same text.

        A quantity may be composed of multiple parts, e.g.
        a number and a unit, and therefore correspond to
        multiple CoreNLP tokens.

        Parameters
        ----------
        quantity : dict
            A Grobid quantity.
        tokens : list
            CoreNLP tokens.

        Returns
        -------
        ids : list
            A list of CoreNLP token IDs corresponding to
            the given quantity.
        """
        value_start = quantity["offsetStart"]
        value_end = quantity["offsetEnd"]
        ids = self.get_overlapping_token_ids(value_start, value_end, tokens)
        if "rawUnit" in quantity:
            unit_start = quantity["rawUnit"]["offsetStart"]
            unit_end = quantity["rawUnit"]["offsetEnd"]
            ids += self.get_overlapping_token_ids(unit_start, unit_end, tokens)

        return ids

    def get_measurement_tokens(self, measurement, tokens):
        """Associate a Grobid measurement to CoreNLP tokens.

        See `get_quantity_tokens` for more details.

        Parameters
        ----------
        measurement : dict
            A Grobid measurement.
        tokens : list
            CoreNLP tokens.

        Returns
        -------
        ids : list
            A list of CoreNLP token IDs corresponding to
            the given quantity.
        """
        ids = []

        for quantity in self.iter_quantities(measurement):
            ids += self.get_quantity_tokens(quantity, tokens)

        return ids

    def get_entity_tokens(self, entity, tokens):
        """Associate a spacy entity to CoreNLP tokens.

        Parameters
        ----------
        entity : spacy.tokens.Span
            A spacy entity extracted from the text. See
            `extract_attributes` for more details.
        tokens : list
            CoreNLP tokens.

        Returns
        -------
        ids : list
            A list of CoreNLP token IDs corresponding to
            the given entity.
        """
        return self.get_overlapping_token_ids(
            entity.start_char,
            entity.end_char,
            tokens)

    @staticmethod
    def find_compound_parents(dependencies, tokens_d, token_idx):
        """Parse CoreNLP dependencies to find parents of token.

        To link named entities to attributes parents for both
        entity tokens and attribute tokens need to be extracted.
        See `extract_attributes` for more information

        This is one possible strategy for finding parents of
        a given token. For a given entity find direct
        parents with the relation type "compound".

        Parameters
        ----------
        dependencies : list
            CoreNLP dependencies found in
            response['sentences'][idx][['basicDependencies']
        tokens_d : dict
            CoreNLP token dictionary mapping token indices
            to tokens. See `extract_attributes`.
        token_idx : int
            The index of the token for which parents
            need to be found.

        Returns
        -------
        parents : list
            A list of parents.
        """
        parents = []
        for link in dependencies:
            if link['dependent'] == token_idx and link['dep'] == "compound":
                parents.append(link['governor'])

        return parents

    @staticmethod
    def iter_parents(dependencies, token_idx):
        """Iterate over all parents of a token.

        It seems that each node has at most one parent, and
        that `parent == 0` means no parent

        Parameters
        ----------
        dependencies : list
            CoreNLP dependencies found in
            response['sentences'][idx][['basicDependencies'].
        token_idx : int
            The index of the token for which parents
            need to be iterated.

        Yields
        ------
        parent_idx : int
            The index of a parent token.
        """
        for link in dependencies:
            if link['dependent'] == token_idx:
                parent = link['governor']
                if parent != 0:
                    yield link['governor']

    def find_nn_parents(self, dependencies, tokens_d, token_idx):
        """Parse CoreNLP dependencies to find parents of token.

        To link named entities to attributes parents for both
        entity tokens and attribute tokens need to be extracted.
        See `extract_attributes` for more information

        This is one possible strategy for finding parents of
        a given token. Ascent the dependency tree until find
        a parent of type "NN". Do this for all parents. If, as
        it seems, each node has at most one parent, then
        the results will be either one index or no indices.

        Parameters
        ----------
        dependencies : list
            CoreNLP dependencies found in
            response['sentences'][idx][['basicDependencies']
        tokens_d : dict
            CoreNLP token dictionary mapping token indices
            to tokens. See `extract_attributes`.
        token_idx : int
            The index of the token for which parents
            need to be found.

        Returns
        -------
        parents : list
            A list of parents.
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
        """Find all parents of a given CoreNLP token.

        Parameters
        ----------
        dependencies : list
            CoreNLP dependencies found in
            response['sentences'][idx][['basicDependencies']
        tokens_d : dict
            CoreNLP token dictionary mapping token indices
            to tokens. See `extract_attributes`.
        tokens : list
            List of token indices for which parents
            need to be found.
        parent_fn : function
            An implementation of a parent finding strategy. Currently
            the available strategies are `find_compound_parents` and
            `find_nn_parents`. The latter seems to perform better.

        Returns
        -------
        parent_ids : list
            A list of all parents found under the given strategy for
            the tokens provided.
        """
        if parent_fn is None:
            parent_fn = self.find_nn_parents

        parent_ids = []

        for token_idx in tokens:
            parent_ids += parent_fn(dependencies, tokens_d, token_idx)

        return parent_ids

    def extract_attributes(self, text, linked_attributes_only=True):
        """Extract attributes from text.

        Parameters
        ----------
        text : str
            The text for attribute extraction.
        linked_attributes_only : bool
            If true then only those attributes will be recorded
            for which there is an associated named entity.

        Returns
        -------
        df : pd.DataFrame
            A pandas data frame with extracted attributes.
        """
        # NER
        doc = self.ee_model(text)
        sent = list(doc.sents)[0]
        detected_entities = [ent for ent in sent.ents]
        logging.info("{} entities detected: {}".format(len(detected_entities), detected_entities))

        # Grobid Quantities
        measurements = self.get_grobid_measurements(text)
        logging.info("{} measurements detected".format(len(measurements)))

        # CoreNLP
        logging.info("Sending CoreNLP query...")
        response_json = None
        try:
            request_data = text.encode("utf-8")
            response = requests.post(self.core_nlp_url, data=request_data)
            assert response.status_code == 200
            response_json = json.loads(response.text)
        except requests.exceptions.RequestException:
            warnings.warn("There was a problem contacting the CoreNLP server.")
        except AssertionError:
            warnings.warn("Reply by CoreNLP was not OK.")
        except json.JSONDecodeError:
            warnings.warn("Could not parse the CoreNLP response JSON.")
        finally:
            if response_json is None:
                response_json = {'sentences': []}
        logging.info("CoreNLP found {} sentences".format(len(response_json['sentences'])))

        # Analysis
        columns = ["entity", "entity_type", "attribute"]
        rows = []
        recorded_measurements = set()

        for entity in detected_entities:
            for i, measurement in enumerate(measurements):
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
                        recorded_measurements.add(i)

        if not linked_attributes_only:
            for i, measurement in enumerate(measurements):
                if i not in recorded_measurements:
                    row = {
                        "attribute": measurement
                    }
                    rows.append(row)

        df_attributes = pd.DataFrame(rows, columns=columns)

        return df_attributes

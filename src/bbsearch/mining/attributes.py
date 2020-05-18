import collections
import json
import logging
import warnings

import requests
from IPython.display import HTML


logger = logging.getLogger(__name__)


class AttributeExtractor:

    def __init__(self, core_nlp_url, grobid_quantities_url):
        logger.debug("{} init".format(self.__class__.__name__))
        logger.debug("CoreNLP URL: {}".format(core_nlp_url))
        logger.debug("Grobid Quantities URL: {}".format(grobid_quantities_url))

        self.core_nlp_url = core_nlp_url
        self.grobid_quantities_url = grobid_quantities_url

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
        potential_quantity_keys = [
            'quantity', 'quantityMost', 'quantityLeast',
            'quantities', 'quantityBase', 'quantityRange']
        found_keys = filter(lambda key: key in measurement,
                            potential_quantity_keys)
        key = next(iter(found_keys))

        if key == 'quantities':
            quantity_types = [self.get_quantity_type(quantity) for quantity in
                              measurement['quantities']]
            assert len(set(quantity_types)) == 1
            quantity_type = quantity_types[0]
        else:
            quantity_type = self.get_quantity_type(measurement[key])

        return quantity_type

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

        annotations = []

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

        for measurement in measurements:
            if 'quantity' in measurement:
                annotations += annotate_quantity(measurement['quantity'])
            elif 'quantities' in measurement:
                for quantity in measurement['quantities']:
                    annotations += annotate_quantity(quantity)
            elif 'quantityMost' in measurement or 'quantityLeast' in measurement:
                if 'quantityLeast' in measurement:
                    annotations += annotate_quantity(measurement['quantityLeast'])
                if 'quantityMost' in measurement:
                    annotations += annotate_quantity(measurement['quantityMost'])
            elif 'quantityBase' in measurement or 'quantityRange' in measurement:
                if 'quantityBase' in measurement:
                    annotations += annotate_quantity(measurement['quantityBase'])
                if 'quantityRange' in measurement:
                    annotations += annotate_quantity(measurement['quantityRange'])
            else:
                warnings.warn("no quantity in measurement")
                print(measurement)

        sorted(annotations, key=lambda x: x[0])
        annotated_text = ''
        last_idx = 0
        for start, end, quantity in annotations:
            annotated_text += text[last_idx:start] + quantity
            last_idx = end
        annotated_text += text[last_idx:]
        html = css_styles + f"<div class=\"fixedWidth\">" + annotated_text + "</div>"

        return HTML(html)

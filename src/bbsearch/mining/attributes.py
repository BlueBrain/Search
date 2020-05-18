import collections
import json
import logging
import warnings

import requests


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

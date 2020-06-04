import json
import requests

from bbsearch.searcher import Searcher


class RemoteSearcher(Searcher):

    def __init__(self, search_server_url):
        self.search_server_url = search_server_url

    def query(self,
              which_model,
              k,
              query_text,
              has_journal=False,
              date_range=None,
              deprioritize_strength='None',
              exclusion_text=None,
              deprioritize_text=None,
              verbose=True):
        payload = dict(
            which_model=which_model,
            k=k,
            query_text=query_text,
            has_journal=has_journal,
            date_range=date_range,
            deprioritize_strength=deprioritize_strength,
            exclusion_text=exclusion_text,
            deprioritize_text=deprioritize_text,
            verbose=False,
        )

        response = requests.post(self.search_server_url, json=payload)
        if response.ok:
            response_json = json.loads(response.text)
            sentence_ids = response_json["sentence_ids"]
            similarities = response_json["similarities"]
            stats = response_json["stats"]
        else:
            sentence_ids = None
            similarities = None
            stats = None

        return sentence_ids, similarities, stats

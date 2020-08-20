"""Remote BBS Searcher."""
import requests


class RemoteSearcher:
    """The remote BBS searcher.

    Parameters
    ----------
    search_server_url : str
        The URL of the remote search server.
    """

    def __init__(self, search_server_url):
        self.search_server_url = search_server_url

    def query(
        self,
        which_model,
        k,
        query_text,
        has_journal=False,
        date_range=None,
        deprioritize_strength="None",
        exclusion_text=None,
        inclusion_text=None,
        deprioritize_text=None,
        verbose=True,
    ):
        """Do the search.

        Parameters
        ----------
        which_model : str
            The name of the model to use.
        k : int
            Number of top results to display.
        query_text : str
            Query.
        has_journal : bool
            If True, only consider papers that have a journal information.
        date_range : tuple
            Tuple of form (start_year, end_year) representing the considered
            time range.
        deprioritize_text : str
            Text query of text to be deprioritized.
        deprioritize_strength : str, {'None', 'Weak', 'Mild', 'Strong', 'Stronger'}
            How strong the deprioritization is.
        exclusion_text : str
            New line separated collection of strings that are automatically used to exclude a given sentence.
            If a sentence contains any of these strings then we filter it out.
        inclusion_text : str
            New line separated collection of strings. Only sentences that contain all of these
            strings are going to make it through the filtering.
        verbose : bool
            If True, then printing statistics to standard output.
        """
        payload = dict(
            which_model=which_model,
            k=k,
            query_text=query_text,
            has_journal=has_journal,
            date_range=date_range,
            deprioritize_strength=deprioritize_strength,
            exclusion_text=exclusion_text,
            inclusion_text=inclusion_text,
            deprioritize_text=deprioritize_text,
            verbose=False,
        )

        response = requests.post(self.search_server_url, json=payload)
        if response.ok:
            response_json = response.json()
            sentence_ids = response_json["sentence_ids"]
            similarities = response_json["similarities"]
            stats = response_json["stats"]
        else:
            sentence_ids = None
            similarities = None
            stats = None

        return sentence_ids, similarities, stats

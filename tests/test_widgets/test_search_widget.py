from copy import copy
from unittest.mock import Mock

import numpy as np
import pytest

from bbsearch.search import LocalSearcher
from bbsearch.widgets import ArticleSaver, SearchWidget


class SearchWidgetBot:
    """Bot that interacts with the SearchWidget.

    Attributes
    ----------
    search_widget : SearchWidget
        Instance of the SearchWidget.

    capsys : pytest.fixture
        Captures standard output. It will enable us to capture print statements done by the
        search widget.

    monkeypatch : pytest.fixture
        Allows for easy patching. Note that we patch the `display` function of IPython.
        This way we are going to be able to capture all the objects the
        `search_widget.widgets['out']` holds.

    n_displays_per_article : int
        Number of displayed objects for each article in the top results. Note that
        currently it is 4 since we are outputing:

            - Article metadata : ``IPython.core.display.HTML``
            - Store paragraph checkbox : ``widgets.Checkbox``
            - Store article checkbox : ``widgets.Checkbox``
            - Formatted output of type ``IPython.core.display.HTML``

    """

    def __init__(self, search_widget, capsys, monkeypatch, n_displays_per_article=4):
        self.search_widget = search_widget
        self._display_cached = []
        self._capsys = capsys
        self.n_displays_per_article = n_displays_per_article

        monkeypatch.setattr('bbsearch.widgets.search_widget.display',
                            lambda x: self._display_cached.append(x))

    @property
    def display_cached(self):
        """Return list of displayed objects and clear history."""
        dc_copy = copy(self._display_cached)
        del self._display_cached[:]
        return dc_copy

    @property
    def stdout_cached(self):
        """Return string of standard output and clear history"""
        return self._capsys.readouterr().out

    def click(self, widget_name):
        """Click a widget.

        Parameters
        ----------
        widget_name : str
            Name of the widget.
        """
        self.search_widget.widgets[widget_name].click()

    def set_value(self, widget_name, value):
        """Set a value of a chosen widget.

        Note that this works with multiple different widgets like sliders, dropdowns, ...

        Parameters
        ----------
        widget_name : str
            Name of the widget.

        value : Any
            Value to set the widget to. The type depends on the widget.
        """
        self.search_widget.widgets[widget_name].value = value


@pytest.mark.parametrize('query_text', ['HELLO'])
@pytest.mark.parametrize('k', [3, 5])
@pytest.mark.parametrize('results_per_page', [1, 2, 3])
def test_dummy(fake_sqlalchemy_engine, monkeypatch, capsys, query_text, k, results_per_page):
    """"""
    n_dim = 3
    n_sentences = fake_sqlalchemy_engine.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]

    embedding_model = Mock()
    embedding_model.embed.return_value = np.random.random(n_dim)

    embedding_models = {'BSV': embedding_model}
    precomputed_embeddings = {'BSV': np.random.random((n_sentences, n_dim))}
    indices = np.arange(1, n_sentences + 1)

    searcher = LocalSearcher(embedding_models, precomputed_embeddings, indices, connection=fake_sqlalchemy_engine)

    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=results_per_page)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    # Initial state
    assert 'Click "Investigate" to display some results.' in bot.stdout_cached
    assert not bot.display_cached

    bot.set_value('top_results', k)
    bot.set_value('query_text', query_text)
    bot.click('investigate_button')
    assert len(bot.display_cached) == min(results_per_page, k) * bot.n_displays_per_article

    articles_left = k - min(results_per_page, k)

    # Make sure paging works
    while articles_left > 0:
        bot.click('page_forward')
        displayed_articles = min(results_per_page, articles_left)

        assert len(bot.display_cached) == displayed_articles * bot.n_displays_per_article

        articles_left -= displayed_articles

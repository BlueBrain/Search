import contextlib
import os
import sys
import textwrap
from copy import copy
from pathlib import Path
from unittest.mock import Mock

import ipywidgets
import numpy as np
import pytest
from IPython.display import HTML

from bbsearch.search import LocalSearcher
from bbsearch.widgets import ArticleSaver, SearchWidget
from bbsearch.widgets.search_widget import _Save


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

    n_displays_per_result : int
        Number of displayed objects for each result in the top results. Note that
        currently it is 4 since we are outputing:

            - Article metadata : ``IPython.core.display.HTML``
            - Store paragraph checkbox : ``widgets.Checkbox``
            - Store article checkbox : ``widgets.Checkbox``
            - Formatted output of type ``IPython.core.display.HTML``

    """

    def __init__(self, search_widget, capsys, monkeypatch, n_displays_per_result=4):
        self.search_widget = search_widget
        self._display_cached = []
        self._capsys = capsys
        self.n_displays_per_result = n_displays_per_result

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

    def get_value(self, widget_name):
        """Get a value of a chosen widget.

        Parameters
        ----------
        widget_name : str
            Name of the widget.

        Returns
        -------
        value : Any
            Current value of the widget.
        """
        return self.search_widget.widgets[widget_name].value

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


@contextlib.contextmanager
def cd_temp(path):
    """Change working directory and return to previous on exit.

    Parameters
    ----------
    path : str or Path
        Path to the directory.
    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def create_searcher(engine, n_dim=2):
    """Create a LocalSearcher in some reasonable way."""
    n_sentences = engine.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]

    embedding_model = Mock()
    embedding_model.embed.return_value = np.random.random(n_dim)

    embedding_models = {'BSV': embedding_model}
    precomputed_embeddings = {'BSV': np.random.random((n_sentences, n_dim))}
    indices = np.arange(1, n_sentences + 1)

    searcher = LocalSearcher(embedding_models, precomputed_embeddings, indices, connection=engine)
    return searcher


@pytest.mark.parametrize('query_text', ['HELLO'])
@pytest.mark.parametrize('k', [3, 5])
@pytest.mark.parametrize('results_per_page', [1, 2, 3])
def test_paging(fake_sqlalchemy_engine, monkeypatch, capsys, query_text, k, results_per_page):
    """Test that paging is displaying the right number results"""

    searcher = create_searcher(fake_sqlalchemy_engine)
    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=results_per_page)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    # Initial state
    assert 'Click on "Search Literature!" button to display some results.' in bot.stdout_cached
    assert not bot.display_cached

    bot.set_value('top_results', k)
    bot.set_value('query_text', query_text)
    bot.click('investigate_button')
    assert len(bot.display_cached) == min(results_per_page, k) * bot.n_displays_per_result

    results_left = k - min(results_per_page, k)

    # Make sure paging works
    while results_left > 0:
        bot.click('page_forward')
        displayed_results = min(results_per_page, results_left)

        assert len(bot.display_cached) == displayed_results * bot.n_displays_per_result

        results_left -= displayed_results


def test_correct_results_order(fake_sqlalchemy_engine, monkeypatch, capsys):
    """Check that the most relevant sentence is the first result."""
    n_sentences = fake_sqlalchemy_engine.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]

    most_relevant_bsv_id = 7
    query_bsv = f'SELECT text FROM sentences WHERE sentence_id = {most_relevant_bsv_id}'
    most_relevant_bsv_text = fake_sqlalchemy_engine.execute(query_bsv).fetchone()[0]

    most_relevant_sbiobert_id = 3
    query_sbiobert = f'SELECT text FROM sentences WHERE sentence_id = {most_relevant_sbiobert_id}'
    most_relevant_sbiobert_text = fake_sqlalchemy_engine.execute(query_sbiobert).fetchone()[0]

    embedding_model_bsv = Mock()
    embedding_model_bsv.embed.return_value = np.array([0, 1])  # 90 degrees
    embedding_model_sbiobert = Mock()
    embedding_model_sbiobert.embed.return_value = np.array([0, -1])  # 270 degrees

    embedding_models = {'BSV': embedding_model_bsv,
                        'SBioBERT': embedding_model_sbiobert}

    precomputed_embeddings = {'BSV': np.ones((n_sentences, 2)),  # 45 degrees
                              'SBioBERT': np.ones((n_sentences, 2))}  # 45 degrees

    precomputed_embeddings['BSV'][most_relevant_bsv_id - 1] = np.array([0.1, 0.9])  # ~90 degrees
    precomputed_embeddings['SBioBERT'][most_relevant_sbiobert_id - 1] = np.array([0.1, -0.9])  # ~270 degrees

    indices = np.arange(1, n_sentences + 1)

    searcher = LocalSearcher(embedding_models,
                             precomputed_embeddings,
                             indices,
                             connection=fake_sqlalchemy_engine)

    k = 1
    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=k)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('top_results', k)
    bot.set_value('print_paragraph', False)

    # BSV
    bot.set_value('sent_embedder', 'BSV')
    bot.click('investigate_button')

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result
    assert textwrap.fill(most_relevant_bsv_text, width=80) in captured_display_objects[-1].data

    # SBioBERT
    bot.set_value('sent_embedder', 'SBioBERT')
    bot.click('investigate_button')

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result
    assert textwrap.fill(most_relevant_sbiobert_text, width=80) in captured_display_objects[-1].data


@pytest.mark.parametrize('saving_mode', [_Save.NOTHING, _Save.PARAGRAPH, _Save.ARTICLE])
def test_article_saver_gets_updated(fake_sqlalchemy_engine, monkeypatch, capsys, saving_mode):
    """When clicking the paragraph or article checkbox the ArticleSaver state is modified."""
    searcher = create_searcher(fake_sqlalchemy_engine)
    k = 10
    result_to_take = 3

    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=k)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('top_results', k)
    bot.set_value('default_value_article_saver', _Save.NOTHING)
    bot.click('investigate_button')

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result
    assert bot.get_value('default_value_article_saver') == _Save.NOTHING

    start = result_to_take * bot.n_displays_per_result
    end = (result_to_take + 1) * bot.n_displays_per_result
    meta, chb_paragraph, chb_article, out = captured_display_objects[start: end]

    # Check the checkbox
    if saving_mode == _Save.NOTHING:

        assert not widget.article_saver.state

    elif saving_mode == _Save.PARAGRAPH:
        chb_paragraph.value = True

        assert len(widget.article_saver.state) == 1  # actual len is 0
        assert list(widget.article_saver.state)[0][1] != -1

    elif saving_mode == _Save.ARTICLE:
        chb_article.value = True

        assert len(widget.article_saver.state) == 1
        assert list(widget.article_saver.state)[0][1] == -1  # actual value 4

    else:
        raise ValueError(f'Unrecognized saving mode: {saving_mode}')


@pytest.mark.parametrize('saving_mode', [_Save.NOTHING, _Save.PARAGRAPH, _Save.ARTICLE])
def test_article_saver_global(fake_sqlalchemy_engine, monkeypatch, capsys, saving_mode):
    """Make sure that default saving buttons result in correct checkboxes."""

    searcher = create_searcher(fake_sqlalchemy_engine)
    k = 10

    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=k)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('top_results', k)
    bot.set_value('default_value_article_saver', saving_mode)
    bot.click('investigate_button')

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result

    if saving_mode == _Save.NOTHING:
        assert not widget.article_saver.state

    elif saving_mode == _Save.PARAGRAPH:
        assert 0 < len(widget.article_saver.state) <= k
        assert all(x[1] != -1 for x in widget.article_saver.state)

    elif saving_mode == _Save.ARTICLE:
        assert 0 < len(widget.article_saver.state) <= k
        assert all(x[1] == -1 for x in widget.article_saver.state)
    else:
        raise ValueError(f'Unrecognized saving mode: {saving_mode}')

    for i, display_obj in enumerate(captured_display_objects):
        if isinstance(display_obj, ipywidgets.Checkbox):
            if display_obj.description == 'Extract the paragraph':
                assert display_obj.value == (saving_mode == _Save.PARAGRAPH)

            elif display_obj.description == 'Extract the entire article':
                assert display_obj.value == (saving_mode == _Save.ARTICLE)

            else:
                raise ValueError(f'Unrecognized checkbox, {i}')

        elif isinstance(display_obj, HTML):
            pass

        else:
            raise TypeError(f'Unrecognized type: {type(display_obj)}')


def test_inclusion_text(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    searcher = create_searcher(fake_sqlalchemy_engine)
    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine),
                          results_per_page=10)

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('inclusion_text', "")
    bot.click('investigate_button')

    assert bot.display_cached

    bot.set_value('inclusion_text', "THIS TEXT DOES NOT EXIST IN ANY SENTENCE")
    bot.click('investigate_button')

    assert not bot.display_cached


@pytest.mark.skipif(sys.platform != "darwin", reason="Bug in wkhtmltopdf")
def test_pdf(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    """Make sure creation of PDF report works."""
    tmpdir = Path(tmpdir)
    searcher = create_searcher(fake_sqlalchemy_engine)

    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine))

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('top_results', 2)
    bot.click('investigate_button')

    bot.stdout_cached  # clear standard output

    with cd_temp(tmpdir):
        bot.click('report_button')

    assert 'Creating the search results PDF report...' in bot.stdout_cached

    assert len([f for f in tmpdir.iterdir() if f.suffix == '.pdf']) == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="Bug in wkhtmltopdf")
def test_pdf_article_saver(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    """Make sure creation of PDF article saver state works."""
    tmpdir = Path(tmpdir)
    searcher = create_searcher(fake_sqlalchemy_engine)

    widget = SearchWidget(searcher,
                          fake_sqlalchemy_engine,
                          ArticleSaver(fake_sqlalchemy_engine))

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value('top_results', 2)
    bot.set_value('default_value_article_saver', _Save.ARTICLE)
    bot.click('investigate_button')

    bot.stdout_cached  # clear standard output

    with cd_temp(tmpdir):
        bot.click('articles_button')

    assert 'Creating the saved results PDF report... ' in bot.stdout_cached

    assert len([f for f in tmpdir.iterdir() if f.suffix == '.pdf']) == 1

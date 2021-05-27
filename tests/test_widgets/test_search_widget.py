"""Tests covering the search widget."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import contextlib
import json
import os
import textwrap
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, List
from unittest.mock import Mock

import ipywidgets
import numpy as np
import pytest
import responses
import torch
from IPython.display import HTML

from bluesearch.search import SearchEngine
from bluesearch.widgets import ArticleSaver, SearchWidget
from bluesearch.widgets.search_widget import _Save


class SearchWidgetBot:
    """Bot that interacts with the SearchWidget.

    Attributes
    ----------
    search_widget : SearchWidget
        Instance of the SearchWidget.

    capsys : pytest.fixture
        Captures standard output. It will enable us to capture print
        statements done by the search widget.

    monkeypatch : pytest.fixture
        Allows for easy patching. Note that we patch the `display`
        function of IPython. This way we are going to be able to
        capture all the objects the `search_widget.widgets['out']` holds.

    n_displays_per_result : int
        Number of displayed objects for each result in the top results.
        Note that currently it is 4 since we are outputting:

            - Article metadata : ``IPython.core.display.HTML``
            - Store paragraph checkbox : ``widgets.Checkbox``
            - Store article checkbox : ``widgets.Checkbox``
            - Formatted output of type ``IPython.core.display.HTML``

    """

    def __init__(self, search_widget, capsys, monkeypatch, n_displays_per_result=4):
        self.search_widget = search_widget
        self._display_cached: List[Any] = []
        self._capsys = capsys
        self.n_displays_per_result = n_displays_per_result

        monkeypatch.setattr(
            "bluesearch.widgets.search_widget.display",
            lambda x: self._display_cached.append(x),
        )

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

        Note that this works with multiple different widgets like sliders,
        dropdowns, ...

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
    n_sentences = engine.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]

    embedding_model = Mock()
    embedding_model.embed.return_value = np.random.random(n_dim)

    embedding_models = {"SBioBERT": embedding_model}
    embeddings = torch.rand((n_sentences, n_dim)).to(dtype=torch.float32)
    norm = torch.norm(input=embeddings, dim=1, keepdim=True)
    norm[norm == 0] = 1
    embeddings /= norm
    precomputed_embeddings = {"SBioBERT": embeddings}
    indices = np.arange(1, n_sentences + 1)

    searcher = SearchEngine(
        embedding_models, precomputed_embeddings, indices, connection=engine
    )
    return searcher


def activate_responses(fake_sqlalchemy_engine):
    searcher = create_searcher(fake_sqlalchemy_engine)
    http_address = "http://test"
    responses.add_callback(
        responses.POST,
        http_address,
        callback=partial(request_callback, searcher=searcher),
        content_type="application/json",
    )
    return http_address


def request_callback(request, searcher):
    payload = json.loads(request.body)
    top_sentence_ids, top_similarities, stats = searcher.query(**payload)
    headers = {"request-id": "1234abcdeABCDE"}
    resp_body = {
        "sentence_ids": top_sentence_ids.tolist(),
        "similarities": top_similarities.tolist(),
        "stats": stats,
    }
    response = (200, headers, json.dumps(resp_body))
    return response


def request_callback_help(request):
    resp_body = {
        "database": "test_database",
        "supported_models": ["SBioBERT"],
        "version": "1.2.3",
    }
    headers = {"request-id": "1234abcdeABCDE"}
    response = (200, headers, json.dumps(resp_body))
    return response


@responses.activate
@pytest.mark.parametrize("query_text", ["HELLO"])
@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("results_per_page", [1, 2, 3])
def test_paging(
    fake_sqlalchemy_engine, monkeypatch, capsys, query_text, k, results_per_page
):
    """Test that paging is displaying the right number results"""

    http_address = activate_responses(fake_sqlalchemy_engine)

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(connection=fake_sqlalchemy_engine),
        results_per_page=results_per_page,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    # Initial state
    assert (
        'Click on "Search Literature!" button to display some results.'
        in bot.stdout_cached
    )
    assert not bot.display_cached

    bot.set_value("top_results", k)
    bot.set_value("granularity", "sentences")
    bot.set_value("query_text", query_text)
    bot.click("investigate_button")
    assert (
        len(bot.display_cached) == min(results_per_page, k) * bot.n_displays_per_result
    )

    results_left = k - min(results_per_page, k)

    # Make sure paging works
    while results_left > 0:
        bot.click("page_forward")
        displayed_results = min(results_per_page, results_left)

        assert len(bot.display_cached) == displayed_results * bot.n_displays_per_result

        results_left -= displayed_results


@responses.activate
def test_correct_results_order(fake_sqlalchemy_engine, monkeypatch, capsys):
    """Check that the most relevant sentence is the first result."""
    n_sentences = fake_sqlalchemy_engine.execute(
        "SELECT COUNT(*) FROM sentences"
    ).fetchone()[0]

    most_relevant_sbiobert_id = 7
    query_sbiobert = (
        f"SELECT text FROM sentences WHERE sentence_id = {most_relevant_sbiobert_id}"
    )
    most_relevant_sbiobert_text = fake_sqlalchemy_engine.execute(
        query_sbiobert
    ).fetchone()[0]

    embedding_model_sbiobert = Mock()
    embedding_model_sbiobert.embed.return_value = np.array([0, 1])  # 90 degrees

    embedding_models = {
        "SBioBERT": embedding_model_sbiobert,
    }

    precomputed_embeddings = {
        "SBioBERT": torch.ones((n_sentences, 2)).to(dtype=torch.float32)
        / 2 ** (1 / 2),  # 45 degrees
    }

    norm = (0.1 ** 2 + 0.9 ** 2) ** (1 / 2)
    precomputed_embeddings["SBioBERT"][most_relevant_sbiobert_id - 1, :] = (
        torch.tensor([0.1, 0.9]) / norm
    )
    # ~90 degrees

    indices = np.arange(1, n_sentences + 1)

    searcher = SearchEngine(
        embedding_models,
        precomputed_embeddings,
        indices,
        connection=fake_sqlalchemy_engine,
    )

    responses.add_callback(
        responses.POST,
        "http://test",
        callback=partial(request_callback, searcher=searcher),
        content_type="application/json",
    )

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    k = 1
    widget = SearchWidget(
        bbs_search_url="http://test",
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
        results_per_page=k,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("top_results", k)
    bot.set_value("print_paragraph", False)

    bot.set_value("sent_embedder", "SBioBERT")
    bot.click("investigate_button")

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result
    assert (
        textwrap.fill(most_relevant_sbiobert_text, width=80)
        in captured_display_objects[-1].data
    )


@responses.activate
@pytest.mark.parametrize("saving_mode", [_Save.NOTHING, _Save.PARAGRAPH, _Save.ARTICLE])
def test_article_saver_gets_updated(
    fake_sqlalchemy_engine, monkeypatch, capsys, saving_mode
):
    """Clicking paragraph or article checkbox modifies the ArticleSaver state."""

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    k = 10
    result_to_take = 3

    http_address = activate_responses(fake_sqlalchemy_engine)

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
        results_per_page=k,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("top_results", k)
    bot.set_value("default_value_article_saver", _Save.NOTHING)
    bot.click("investigate_button")

    captured_display_objects = bot.display_cached

    assert len(captured_display_objects) == k * bot.n_displays_per_result
    assert bot.get_value("default_value_article_saver") == _Save.NOTHING

    start = result_to_take * bot.n_displays_per_result
    end = (result_to_take + 1) * bot.n_displays_per_result
    meta, chb_paragraph, chb_article, out = captured_display_objects[start:end]

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
        raise ValueError(f"Unrecognized saving mode: {saving_mode}")


def test_errors(fake_sqlalchemy_engine, monkeypatch, capsys):
    """Check that widget raises an error when bbs search server not working."""

    with pytest.raises(Exception):
        SearchWidget(
            bbs_search_url="fake_address",
            bbs_mysql_engine=fake_sqlalchemy_engine,
            article_saver=ArticleSaver(fake_sqlalchemy_engine),
            results_per_page=3,
        )


@responses.activate
@pytest.mark.parametrize("saving_mode", [_Save.NOTHING, _Save.PARAGRAPH, _Save.ARTICLE])
def test_article_saver_global(fake_sqlalchemy_engine, monkeypatch, capsys, saving_mode):
    """Make sure that default saving buttons result in correct checkboxes."""
    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    k = 10
    http_address = activate_responses(fake_sqlalchemy_engine)

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
        results_per_page=k,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("top_results", k)
    bot.set_value("default_value_article_saver", saving_mode)
    bot.click("investigate_button")

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
        raise ValueError(f"Unrecognized saving mode: {saving_mode}")

    for i, display_obj in enumerate(captured_display_objects):
        if isinstance(display_obj, ipywidgets.Checkbox):
            if display_obj.description == "Extract the paragraph":
                assert display_obj.value == (saving_mode == _Save.PARAGRAPH)

            elif display_obj.description == "Extract the entire article":
                assert display_obj.value == (saving_mode == _Save.ARTICLE)

            else:
                raise ValueError(f"Unrecognized checkbox, {i}")

        elif isinstance(display_obj, HTML):
            pass

        else:
            raise TypeError(f"Unrecognized type: {type(display_obj)}")


@responses.activate
def test_inclusion_text(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    http_address = activate_responses(fake_sqlalchemy_engine)

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
        results_per_page=10,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("inclusion_text", "")
    bot.click("investigate_button")

    assert bot.display_cached

    bot.set_value("inclusion_text", "THIS TEXT DOES NOT EXIST IN ANY SENTENCE")
    bot.click("investigate_button")

    assert not bot.display_cached


@responses.activate
def test_make_report(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    """Make sure creation of report works."""
    tmpdir = Path(tmpdir)
    http_address = activate_responses(fake_sqlalchemy_engine)

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=(fake_sqlalchemy_engine),
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("top_results", 2)
    bot.click("investigate_button")

    bot.stdout_cached  # clear standard output

    with cd_temp(tmpdir):
        bot.click("report_button")

    assert "Creating the search results report..." in bot.stdout_cached

    assert len([f for f in tmpdir.iterdir() if f.suffix == ".html"]) == 1


@responses.activate
def test_report_article_saver(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    """Make sure creation of report with article saver state works."""
    tmpdir = Path(tmpdir)
    http_address = activate_responses(fake_sqlalchemy_engine)

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    bot.set_value("top_results", 2)
    bot.set_value("default_value_article_saver", _Save.ARTICLE)
    bot.click("investigate_button")

    bot.stdout_cached  # clear standard output

    with cd_temp(tmpdir):
        bot.click("articles_button")

    assert "Creating the saved results report... " in bot.stdout_cached

    assert len([f for f in tmpdir.iterdir() if f.suffix == ".html"]) == 1


def get_search_widget_bot(
    fake_sqlalchemy_engine, monkeypatch, capsys, checkpoint_path=None
):
    http_address = activate_responses(fake_sqlalchemy_engine)

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    widget = SearchWidget(
        bbs_search_url=http_address,
        bbs_mysql_engine=fake_sqlalchemy_engine,
        article_saver=ArticleSaver(fake_sqlalchemy_engine),
        checkpoint_path=checkpoint_path,
    )

    bot = SearchWidgetBot(widget, capsys, monkeypatch)

    return bot


@responses.activate
def test_saved_results(fake_sqlalchemy_engine, monkeypatch, capsys):
    # Test saving with the default setting of saving entire articles
    bot = get_search_widget_bot(fake_sqlalchemy_engine, monkeypatch, capsys)
    bot.click("investigate_button")
    displayed = bot.display_cached
    # Make sure some results were displayed
    assert len(displayed) > 0
    # For each item in the search history there should be a row in saved results
    saved_results = bot.search_widget.saved_results()
    assert len(saved_results) == len(bot.search_widget.history)
    # Check that no paragraphs were saved
    assert all(value == "" for value in saved_results["Paragraph"])
    # Check that no paragraph position is shown if paragraph is not saved
    assert all(value == "" for value in saved_results["Paragraph #"])

    # Test not saving because article saver is None
    bot = get_search_widget_bot(fake_sqlalchemy_engine, monkeypatch, capsys)
    bot.search_widget.article_saver = None
    bot.click("investigate_button")
    displayed = bot.display_cached
    # Make sure some results were displayed
    assert len(displayed) > 0
    saved_results = bot.search_widget.saved_results()
    assert len(saved_results) == 0

    # Test not saving because article saver is None
    bot = get_search_widget_bot(fake_sqlalchemy_engine, monkeypatch, capsys)
    bot.set_value("default_value_article_saver", _Save.PARAGRAPH)
    bot.click("investigate_button")
    displayed = bot.display_cached
    # Make sure some results were displayed
    assert len(displayed) > 0
    # For each item in the search history there should be a row in saved results
    saved_results = bot.search_widget.saved_results()
    assert len(saved_results) == len(bot.search_widget.history)
    # Check that all paragraphs were saved
    assert all(value != "" for value in saved_results["Paragraph"])
    # Check that the paragraph position is shown if paragraph is saved
    assert all(value != "" for value in saved_results["Paragraph #"])


@responses.activate
def test_save_load_checkpoint(fake_sqlalchemy_engine, monkeypatch, capsys, tmpdir):
    # Test saving with the default setting of saving entire articles
    bot = get_search_widget_bot(
        fake_sqlalchemy_engine, monkeypatch, capsys, checkpoint_path=tmpdir
    )

    # Try saving data, but no results to save
    bot.click("save_button")
    last_displayed = bot.display_cached[-1].data
    assert "ERROR!" in last_displayed
    assert "No articles or paragraphs selected." in last_displayed

    # Click on "investigate"
    bot.click("investigate_button")

    # Try loading data, but no checkpoint was saved there
    bot.click("load_button")
    last_displayed = bot.display_cached[-1].data
    assert "ERROR!" in last_displayed
    assert "No checkpoint file found to load." in last_displayed

    # Now there are some results, so we can save a checkpoint
    bot.click("save_button")
    last_displayed = bot.display_cached[-1].data
    with bot.search_widget.checkpoint_path.open("r") as f:
        data = json.load(f)
    assert {
        tuple(x) for x in data["article_saver_state"]
    } == bot.search_widget.article_saver.state
    assert [
        tuple(x) for x in data["search_widget_history"]
    ] == bot.search_widget.history
    assert data["database_name"] == bot.search_widget.database_name
    assert data["search_server_version"] == bot.search_widget.search_server_version
    assert "DONE" in last_displayed
    assert "Saving search results to disk..." in last_displayed

    # Now there is a checkpoint, so we can load it
    # Note: if the database name or the server name is different, data is loaded
    # but we raise a warning.
    for db_name in ("test_database", "test_database_2"):
        bot.search_widget.database_name = db_name
        del bot.search_widget.article_saver.state
        del bot.search_widget.history
        bot.click("load_button")
        assert {
            tuple(x) for x in data["article_saver_state"]
        } == bot.search_widget.article_saver.state
        assert [
            list(x) for x in data["search_widget_history"]
        ] == bot.search_widget.history

        displayed = bot.display_cached
        if db_name != "test_database":
            assert "WARNING" in displayed[-1].data
            assert "DONE" in displayed[-2].data
            assert "Loading search results from disk..." in displayed[-2].data
        else:
            assert "DONE" in displayed[-1].data
            assert "Loading search results from disk..." in displayed[-1].data

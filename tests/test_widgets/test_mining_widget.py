"""Tests covering the mining widget."""

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

import json
from copy import copy
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import responses
from IPython.display import HTML

from bluesearch.widgets import ArticleSaver, MiningSchema, MiningWidget


class MiningWidgetBot:
    """Bot that interacts with the MiningWidget.

    Attributes
    ----------
    mining_widget : MiningWidget
        Instance of the MiningWidget.

    capsys : pytest.fixture
        Captures standard output. It will enable us to capture print
        statements done by the mining widget.

    monkeypatch : pytest.fixture
        Allows for easy patching. Note that we patch the `display` function
        of IPython. This way we are going to be able to capture all the
        objects the `mining_widget.widgets['out']` holds.
    """

    def __init__(self, mining_widget, capsys, monkeypatch):
        self.mining_widget = mining_widget
        self._display_cached: List[Any] = []
        self._capsys = capsys

        monkeypatch.setattr(
            "bluesearch.widgets.mining_widget.display",
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
        self.mining_widget.widgets[widget_name].click()

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
        return self.mining_widget.widgets[widget_name].value

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
        self.mining_widget.widgets[widget_name].value = value


columns = {
    "name": pd.Series(["John Smith", "Erica Meyers"]),
    "department": pd.Series(["Accounting", "IT"]),
    "birthday": pd.Series(["November", "March"]),
}
table_extractions = pd.DataFrame(columns)

TESTS_PATH = Path(__file__).resolve().parent.parent


def request_callback(request):
    dataframe = table_extractions.to_csv(index=False)
    resp_body = {"warnings": ["This is a test warning"], "csv_extractions": dataframe}
    headers = {"request-id": "1234abcdeABCDE"}
    response = (200, headers, json.dumps(resp_body))
    return response


def request_callback_help(request):
    resp_body = {"database": "test_database", "version": "1.2.3"}
    headers = {"request-id": "1234abcdeABCDE"}
    response = (200, headers, json.dumps(resp_body))
    return response


@responses.activate
def test_mining_text(monkeypatch, capsys, mining_schema_df):
    mining_schema_df = mining_schema_df.drop_duplicates(ignore_index=True)

    responses.add_callback(
        responses.POST,
        "http://test/text",
        callback=request_callback,
        content_type="application/json",
    )

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    mining_schema = MiningSchema()
    mining_schema.add_from_df(mining_schema_df)
    mining_widget = MiningWidget(
        mining_server_url="http://test",
        mining_schema=mining_schema,
    )

    bot = MiningWidgetBot(mining_widget, capsys, monkeypatch)
    bot.set_value("input_text", "HELLO")
    bot.click("mine_text")

    assert len(responses.calls) == 2

    display_objs = bot.display_cached
    assert len(display_objs) == 3  # 1 schema + 1 warning + 1 table_extractions
    assert isinstance(display_objs[0], pd.DataFrame)

    assert display_objs[0].equals(mining_schema_df)
    assert isinstance(display_objs[1], HTML)
    assert display_objs[2].equals(table_extractions)


@responses.activate
def test_mining_database(monkeypatch, capsys, fake_sqlalchemy_engine, mining_schema_df):
    mining_schema_df = mining_schema_df.drop_duplicates(ignore_index=True)

    responses.add_callback(
        responses.POST,
        "http://test/database",
        callback=request_callback,
        content_type="text/csv",
    )

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    mining_schema = MiningSchema()
    mining_schema.add_from_df(mining_schema_df)
    mining_widget = MiningWidget(
        mining_server_url="http://test",
        mining_schema=mining_schema,
    )
    empty_dataframe = pd.DataFrame()
    assert empty_dataframe.equals(mining_widget.get_extracted_table())

    bot = MiningWidgetBot(mining_widget, capsys, monkeypatch)
    bot.set_value("input_text", "HELLO")
    bot.click("mine_articles")

    assert len(responses.calls) == 1
    assert "No article saver was provided. Nothing to mine." in bot.stdout_cached

    article_saver = ArticleSaver(fake_sqlalchemy_engine)
    for i in range(2):
        article_saver.add_article(article_id=i)

    mining_widget = MiningWidget(
        mining_server_url="http://test",
        mining_schema=mining_schema,
        article_saver=article_saver,
    )

    bot = MiningWidgetBot(mining_widget, capsys, monkeypatch)
    bot.set_value("input_text", "HELLO")
    bot.click("mine_articles")

    assert len(responses.calls) == 3
    assert "Collecting saved items..." in bot.stdout_cached
    assert isinstance(mining_widget.get_extracted_table(), pd.DataFrame)

    display_objs = bot.display_cached
    assert len(display_objs) == 3  # 1 schema + 1 warning + 1 table_extractions
    assert isinstance(display_objs[0], pd.DataFrame)
    assert isinstance(display_objs[2], pd.DataFrame)

    assert display_objs[0].equals(mining_schema_df)
    assert isinstance(display_objs[1], HTML)
    assert display_objs[2].equals(table_extractions)


@responses.activate
def test_save_load_checkpoint(monkeypatch, capsys, mining_schema_df, tmpdir):
    mining_schema_df = mining_schema_df.drop_duplicates(ignore_index=True)
    responses.add_callback(
        responses.POST,
        "http://test/text",
        callback=request_callback,
        content_type="application/json",
    )

    responses.add_callback(
        responses.POST,
        "http://test/help",
        callback=request_callback_help,
        content_type="application/json",
    )

    mining_schema = MiningSchema()
    mining_schema.add_from_df(mining_schema_df)
    mining_widget = MiningWidget(
        mining_server_url="http://test",
        mining_schema=mining_schema,
        checkpoint_path=tmpdir,
    )

    bot = MiningWidgetBot(mining_widget, capsys, monkeypatch)
    bot.set_value("input_text", "HELLO")

    # Try saving data, but no results to save
    bot.click("save_button")
    last_displayed = bot.display_cached[-1].data
    assert "ERROR!" in last_displayed
    assert "No mining results available." in last_displayed

    # Click on "investigate"
    bot.click("mine_text")

    # Try loading data, but no checkpoint was saved there
    bot.click("load_button")
    last_displayed = bot.display_cached[-1].data
    assert "ERROR!" in last_displayed
    assert "No checkpoint file found to load." in last_displayed

    # Now there are some results, so we can save a checkpoint
    bot.click("save_button")
    displayed = bot.display_cached
    with bot.mining_widget.checkpoint_path.open("r") as f:
        data = json.load(f)
    assert np.array_equal(
        pd.DataFrame(data["mining_widget_extractions"]).values,
        bot.mining_widget.table_extractions.values,
    )
    assert data["database_name"] == bot.mining_widget.database_name
    assert data["mining_server_version"] == bot.mining_widget.mining_server_version
    assert "DONE" in displayed[-1].data
    assert "Saving mining results to disk..." in displayed[-2].data

    # Now there is a checkpoint, so we can load it
    # Note: if the database name or the server name is different, data is loaded
    # but we raise a warning.
    for db_name in ("test_database", "test_database_2"):
        bot.mining_widget.database_name = db_name
        del bot.mining_widget.table_extractions
        bot.click("load_button")
        assert np.array_equal(
            pd.DataFrame(data["mining_widget_extractions"]).values,
            bot.mining_widget.table_extractions.values,
        )

        displayed = bot.display_cached
        if db_name != "test_database":
            assert isinstance(displayed[-1], pd.DataFrame)
            assert "WARNING" in displayed[-2].data
            assert "DONE" in displayed[-3].data
            assert "Loading mining results from disk..." in displayed[-4].data
        else:
            assert isinstance(displayed[-1], pd.DataFrame)
            assert "DONE" in displayed[-2].data
            assert "Loading mining results from disk..." in displayed[-3].data

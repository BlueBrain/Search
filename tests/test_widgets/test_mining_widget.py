import json
from copy import copy
from pathlib import Path

import pandas as pd
import responses
from IPython.display import HTML

from bbsearch.widgets import ArticleSaver, MiningSchema, MiningWidget


class MiningWidgetBot:
    """Bot that interacts with the MiningWidget.

    Attributes
    ----------
    mining_widget : MiningWidget
        Instance of the MiningWidget.

    capsys : pytest.fixture
        Captures standard output. It will enable us to capture print statements done by the
        mining widget.

    monkeypatch : pytest.fixture
        Allows for easy patching. Note that we patch the `display` function of IPython.
        This way we are going to be able to capture all the objects the
        `mining_widget.widgets['out']` holds.
    """

    def __init__(self, mining_widget, capsys, monkeypatch):
        self.mining_widget = mining_widget
        self._display_cached = []
        self._capsys = capsys

        monkeypatch.setattr(
            "bbsearch.widgets.mining_widget.display",
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

        Note that this works with multiple different widgets like sliders, dropdowns, ...

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
    resp_body = {"database": "test_database"}
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

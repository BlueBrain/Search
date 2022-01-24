"""Tests covering the handling of identifiers."""

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

from __future__ import annotations

import pytest

from bluesearch.database.article import ArticleParser


class FakeParser(ArticleParser):
    def __init__(self, title="", authors=(), abstract=(), paragraphs=(), doi=None):
        self._title = title
        self._authors = authors
        self._abstract = abstract
        self._paragraphs = paragraphs
        self._doi = doi

    @property
    def title(self):
        return self._title

    @property
    def authors(self):
        yield from self._authors

    @property
    def abstract(self):
        yield from self._abstract

    @property
    def paragraphs(self):
        yield from self._paragraphs

    @property
    def doi(self):
        return self._doi


class TestIdentifiers:
    # By running this test several times and on different platforms during CI,
    # this test checks that UID generation is deterministic across platforms
    # and Python processes.

    @pytest.mark.parametrize(
        "identifiers, expected",
        [
            pytest.param(
                ("a", "b"), "aca14e654bc28ce1c1e8131004244d64", id="all-defined"
            ),
            pytest.param(
                ("b", "a"), "82ca240c4a3f5579a5c33404af58e41b", id="all-defined-reverse"
            ),
            pytest.param(
                ("a", None), "4b515f920fbbc7954fc5a68bb746b109", id="with-none"
            ),
            pytest.param(
                (None, "a"), "77f283f2e87b852ed7a881e6f638aa80", id="with-none-reverse"
            ),
            pytest.param((None, None), None, id="all-none"),
            pytest.param(
                (None, 0), "14536e026b2a39caf27f3da802e7fed6", id="none-and-zero"
            ),
        ],
    )
    def test_generate_uid_from_identifiers(self, identifiers, expected):
        if expected is None:
            with pytest.raises(ValueError):
                ArticleParser.get_uid_from_identifiers(identifiers)

        else:
            result = ArticleParser.get_uid_from_identifiers(identifiers)
            assert result == expected

            # Check determinism.
            result_bis = ArticleParser.get_uid_from_identifiers(identifiers)
            assert result == result_bis

    @pytest.mark.parametrize(
        "parser_kwargs, expected",
        [
            pytest.param(
                {
                    "title": "TITLE",
                    "abstract": ["ABS 1", "ABS 2"],
                    "paragraphs": [("PAR 1", "text 1"), ("PAR 2", "text 2")],
                    "authors": ["AUTH 1", "AUTH 2"],
                },
                "212f772faf801518f8dd9f745a1c94b2",
                id="no-ids-full-text",
            ),
            pytest.param(
                {
                    "title": "TITLE",
                    "abstract": ["ABS 1", "ABS 2"],
                },
                "7229b18916ba8b83b20d243d5caaf56a",
                id="no-ids-abstract-only",
            ),
            pytest.param(
                {"title": "TITLE", "abstract": ["ABS 1", "ABS 2"], "doi": "1.234"},
                "9cfc45f5817b544ac26e02a9071802b6",
                id="doi-with-text",
            ),
            pytest.param(
                {"doi": "1.234"},
                "9cfc45f5817b544ac26e02a9071802b6",
                id="doi-without-text",
            ),
        ],
    )
    def test_general_article_uid(self, parser_kwargs, expected):
        article_parser = FakeParser(**parser_kwargs)

        assert article_parser.uid == expected

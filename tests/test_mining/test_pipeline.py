"""Collection of tests focused on the bluesearch.mining.pipeline module."""

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

from typing import Dict, List, Tuple
from unittest.mock import Mock

import pandas as pd
import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span

from bluesearch.mining import StartWithTheSameLetter, run_pipeline
from bluesearch.mining.relation import REModel


@pytest.mark.parametrize("n_paragraphs", [0, 1, 5])
@pytest.mark.parametrize("debug", [True, False], ids=["debug", "official_spec"])
def test_overall(model_entities, debug, n_paragraphs):
    text = (
        "This is a filler sentence. Britney Spears had a concert in "
        "Brazil yesterday. And I am a filler too."
    )

    # wrong arguments
    with pytest.raises(TypeError):
        run_pipeline([], model_entities, {("etype_1", "etype_2"): ["WRONG TYPE"]})

    # entities are [Britney Spears, Brazil, yesterday]
    doc = model_entities(text)
    ents = list(doc.ents)
    sents = list(doc.sents)
    etypes = [e.label_ for e in ents]

    # Just make sure the the spacy model is the same
    assert isinstance(doc, Doc)

    assert len(ents) == 3
    assert all([isinstance(e, Span) for e in ents])

    assert len(sents) == 3
    assert all([isinstance(s, Span) for s in sents])

    assert etypes == ["PERSON", "GPE", "DATE"]

    models_relations = {
        ("PERSON", "DATE"): [StartWithTheSameLetter()],
        ("PERSON", "GPE"): [StartWithTheSameLetter()],
    }
    texts = n_paragraphs * [(text, {"important_parameter": 10})]
    df = run_pipeline(texts, model_entities, models_relations, debug)

    official_specs = [
        "entity",
        "entity_type",
        "property",
        "property_value",
        "property_type",
        "property_value_type",
        "ontology_source",
        "paper_id",
        "start_char",
        "end_char",
    ]

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_paragraphs * (
        3 + 1 + 1
    )  # 3 entities, 1 ('PERSON', 'DATE') relation and ('PERSON', 'GPE') relation

    if n_paragraphs > 0:
        if debug:
            assert df.columns.to_list() != official_specs
            assert "important_parameter" in df.columns
            # With all(df["important_parameter"] == 10) mypy throws the error:
            # Non-overlapping equality check (left operand type: "Series",
            # right operand type: "Literal[10]") [comparison-overlap]
            assert df["important_parameter"].map(lambda x: x == 10).all()

        else:
            assert df.columns.to_list() == official_specs


@pytest.mark.parametrize("n_paragraphs", [0, 1, 5])
@pytest.mark.parametrize("debug", [True, False], ids=["debug", "official_spec"])
def test_without_relation(model_entities, debug, n_paragraphs):
    text = (
        "This is a filler sentence. Britney Spears had a concert "
        "in Brazil yesterday. And I am a filler too."
    )

    models_relations: Dict[Tuple[str], List[REModel]] = {}
    texts = n_paragraphs * [(text, {"important_parameter": 10})]
    df = run_pipeline(texts, model_entities, models_relations, debug)

    official_specs = [
        "entity",
        "entity_type",
        "property",
        "property_value",
        "property_type",
        "property_value_type",
        "ontology_source",
        "paper_id",
        "start_char",
        "end_char",
    ]

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_paragraphs * 3  # 3 entities

    if n_paragraphs > 0:
        if debug:
            assert df.columns.to_list() != official_specs
            assert "important_parameter" in df.columns
            assert df["important_parameter"].map(lambda x: x == 10).all()

        else:
            assert df.columns.to_list() == official_specs


def test_not_entity_label(model_entities):
    texts = ["California is a state.", {}]

    doc = model_entities(texts[0])  # detects only "California"
    new_ent = Span(doc, start=3, end=4, label="NaE")
    doc.ents = doc.ents + (new_ent,)  # we add "state" as an entity

    model_entities_m = Mock(spec=Language)
    model_entities_m.pipe.return_value = [(doc, {})]
    models_relations: Dict[Tuple[str], List[REModel]] = {}

    df_1 = run_pipeline(
        texts, model_entities_m, models_relations, excluded_entity_type="!"
    )
    assert len(df_1) == 2

    df_2 = run_pipeline(
        texts, model_entities_m, models_relations, excluded_entity_type="NaE"
    )
    assert len(df_2) == 1

    df_3 = run_pipeline(
        texts, model_entities_m, models_relations, excluded_entity_type=None
    )
    assert len(df_3) == 2

    df_4 = run_pipeline(
        texts, model_entities_m, models_relations, excluded_entity_type="GPE"
    )
    assert len(df_4) == 1

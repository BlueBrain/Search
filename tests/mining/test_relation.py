"""Collection of tests focused on the bluesearch.mining.relation module"""

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

from unittest.mock import Mock

import pytest
from spacy.tokens import Doc, Span

from bluesearch.mining import ChemProt, StartWithTheSameLetter, annotate


def test_annotate(model_entities):
    text = (
        "This is a filler sentence. Bill Gates founded Microsoft and "
        "currently lives in the USA."
    )

    # entities are [Bill Gates, Microsoft, USA]
    # etypes are ['PERSON', 'ORG', 'GPE']
    doc = model_entities(text)
    ents = list(doc.ents)
    sents = list(doc.sents)
    etypes = [e.label_ for e in ents]
    etype_symbols = {
        "PERSON": ("<< ", " >>"),
        "ORG": ("[[ ", " ]]"),
        "GPE": ("{{ ", " }}"),
    }

    # Just make sure the the spacy model is the same
    assert isinstance(doc, Doc)

    assert len(ents) == 3
    assert all([isinstance(e, Span) for e in ents])

    assert len(sents) == 2
    assert all([isinstance(s, Span) for s in sents])

    assert etypes == ["PERSON", "ORG", "GPE"]

    # Wrong arguments
    with pytest.raises(ValueError):
        annotate(doc, sents[1], ents[0], ents[0], etype_symbols)  # identical entities

    with pytest.raises(ValueError):
        annotate(
            doc, sents[0], ents[0], ents[1], etype_symbols
        )  # not in the right sentence

    with pytest.raises(ValueError):
        annotate(doc, sents[1], ents[0], ents[1], {})  # missing symbols

    # Actual tests
    res_1 = annotate(doc, sents[1], ents[0], ents[1], etype_symbols)
    res_2 = annotate(doc, sents[1], ents[1], ents[2], etype_symbols)
    res_3 = annotate(doc, sents[1], ents[2], ents[0], etype_symbols)

    true_1 = "<< Bill Gates >> founded [[ Microsoft ]] and currently lives in the USA."
    true_2 = "Bill Gates founded [[ Microsoft ]] and currently lives in the {{ USA }}."
    true_3 = "<< Bill Gates >> founded Microsoft and currently lives in the {{ USA }}."

    assert res_1 == annotate(
        doc, sents[1], ents[1], ents[0], etype_symbols
    )  # symmetric
    assert res_2 == annotate(
        doc, sents[1], ents[2], ents[1], etype_symbols
    )  # symmetric
    assert res_3 == annotate(
        doc, sents[1], ents[0], ents[2], etype_symbols
    )  # symmetric

    assert res_1 == true_1
    assert res_2 == true_2
    assert res_3 == true_3


@pytest.mark.parametrize("return_prob", [True, False])
def test_chemprot(monkeypatch, return_prob):
    """Run only if scibert installed.

    By default this test will be skipped by the CI since `scibert` introduces
    conflicts. However, one can use it locally to test whether the
    `ChemProt` class works as expected.
    """
    pytest.importorskip("allennlp")
    pytest.importorskip("scibert")  # note that the import itself has a side effect

    # Test scibert import side-effect
    from allennlp.models.model import Model

    assert "text_classifier" in Model.list_available()

    # Prepare test
    class_probs = 13 * [0]
    class_probs[7] = 1

    fake_model = Mock()
    fake_model.predict.return_value = {"class_probs": class_probs}

    fake_predictor = Mock()
    fake_predictor.from_path.return_value = fake_model
    monkeypatch.setattr("allennlp.predictors.Predictor", fake_predictor)

    re_model = ChemProt("")

    # Check REModel logic
    annotated_sentence = (
        "The selective << betaAR >> agonist [[ isoproterenol ]] caused an"
        " enhancement of hippocampal CA3 network activity"
    )

    outputs = re_model.predict(annotated_sentence, return_prob)

    # predict
    if return_prob:
        assert outputs[0] == "AGONIST"
        assert outputs[1] == 1
    else:
        assert outputs == "AGONIST"

    # Symbols
    assert re_model.symbols == {"GGP": ("[[ ", " ]]"), "CHEBI": ("<< ", " >>")}


@pytest.mark.parametrize("return_prob", [True, False])
def test_start_with_the_same_letter(return_prob):
    re_model = StartWithTheSameLetter()

    assert re_model.symbols["etype_1"] == ("[[ ", " ]]")
    assert re_model.symbols["whatever"] == ("[[ ", " ]]")

    annotated_sentence_1 = "Our [[ dad ]] walked the [[ Dog ]]."
    annotated_sentence_2 = "Our [[ dad ]] walked the [[ cat ]]."

    outputs_1 = re_model.predict(annotated_sentence_1, return_prob)
    outputs_2 = re_model.predict(annotated_sentence_2, return_prob)

    if return_prob:
        assert len(outputs_1) == 2 and len(outputs_2) == 2

        relation_1 = outputs_1[0]
        relation_2 = outputs_2[0]

        prob_1 = outputs_1[1]
        prob_2 = outputs_2[1]

        assert prob_1 == 1 and prob_2 == 1

    else:
        relation_1 = outputs_1
        relation_2 = outputs_2

    assert relation_1 == "START_WITH_SAME_LETTER"
    assert relation_2 == "START_WITH_DIFFERENT_LETTER"

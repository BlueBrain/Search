"""Collection of tests focused on the bbsearch.mining.relation module"""

from spacy.tokens import Doc, Span
import pytest

from bbsearch.mining import StartWithTheSameLetter, annotate


def test_annotate(model_entities):
    text = 'This is a filler sentence. Bill Gates founded Microsoft and currently lives in the USA.'

    # entities are [Bill Gates, Microsoft, USA]
    # etypes are ['PERSON', 'ORG', 'GPE']
    doc = model_entities(text)
    ents = list(doc.ents)
    sents = list(doc.sents)
    etypes = [e.label_ for e in ents]
    etype_symbols = {'PERSON': ('<< ', ' >>'),
                     'ORG': ('[[ ', ' ]]'),
                     'GPE': ('{{ ', ' }}')
                     }

    # Just make sure the the spacy model is the same
    assert isinstance(doc, Doc)

    assert len(ents) == 3
    assert all([isinstance(e, Span) for e in ents])

    assert len(sents) == 2
    assert all([isinstance(s, Span) for s in sents])

    assert etypes == ['PERSON', 'ORG', 'GPE']

    # Wrong arguments
    with pytest.raises(ValueError):
        annotate(doc, sents[1], ents[0], ents[0], etype_symbols)  # identical entities

    with pytest.raises(ValueError):
        annotate(doc, sents[0], ents[0], ents[1], etype_symbols)  # not in the right sentence

    with pytest.raises(ValueError):
        annotate(doc, sents[1], ents[0], ents[1], {})  # missing symbols

    # Actual tests
    res_1 = annotate(doc, sents[1], ents[0], ents[1], etype_symbols)
    res_2 = annotate(doc, sents[1], ents[1], ents[2], etype_symbols)
    res_3 = annotate(doc, sents[1], ents[2], ents[0], etype_symbols)

    true_1 = '<< Bill Gates >> founded [[ Microsoft ]] and currently lives in the USA.'
    true_2 = 'Bill Gates founded [[ Microsoft ]] and currently lives in the {{ USA }}.'
    true_3 = '<< Bill Gates >> founded Microsoft and currently lives in the {{ USA }}.'

    assert res_1 == annotate(doc, sents[1], ents[1], ents[0], etype_symbols)  # symmetric
    assert res_2 == annotate(doc, sents[1], ents[2], ents[1], etype_symbols)  # symmetric
    assert res_3 == annotate(doc, sents[1], ents[0], ents[2], etype_symbols)  # symmetric

    assert res_1 == true_1
    assert res_2 == true_2
    assert res_3 == true_3


def test_start_with_the_same_letter():
    re_model = StartWithTheSameLetter()

    assert re_model.symbols['etype_1'] == ('[[ ', ' ]]')
    assert re_model.symbols['whatever'] == ('[[ ', ' ]]')

    annotated_sentence_1 = "Our [[ dad ]] walked the [[ Dog ]]."
    annotated_sentence_2 = "Our [[ dad ]] walked the [[ cat ]]."

    assert re_model.predict(annotated_sentence_1)
    assert not re_model.predict(annotated_sentence_2)

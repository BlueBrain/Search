"""Collection of tests focused on the bbsearch.mining.pipeline module."""

import pandas as pd
import pytest
from spacy.tokens import Doc, Span

from bbsearch.mining import StartWithTheSameLetter, run_pipeline


@pytest.mark.parametrize('n_paragraphs', [0, 1, 5])
@pytest.mark.parametrize('debug', [True, False], ids=['debug', 'official_spec'])
def test_overall(model_entities, debug, n_paragraphs):
    text = 'This is a filler sentence. Britney Spears had a concert in Brazil yesterday. And I am a filler too.'

    # wrong arguments
    with pytest.raises(TypeError):
        run_pipeline([], model_entities, {('etype_1', 'etype_2'): ['WRONG TYPE']})

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

    assert etypes == ['PERSON', 'GPE', 'DATE']

    models_relations = {('PERSON', 'DATE'): [StartWithTheSameLetter()],
                        ('PERSON', 'GPE'): [StartWithTheSameLetter()]
                        }
    texts = n_paragraphs * [(text, {'important_parameter': 10})]
    df = run_pipeline(texts, model_entities, models_relations, debug)

    official_specs = ['entity',
                      'entity_type',
                      'property',
                      'property_value',
                      'property_type',
                      'property_value_type',
                      'ontology_source',
                      'paper_id',
                      'start_char',
                      'end_char']

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_paragraphs * (
            3 + 1 + 1)  # 3 entities, 1 ('PERSON', 'DATE') relation and ('PERSON', 'GPE') relation

    if n_paragraphs > 0:
        if debug:
            assert df.columns.to_list() != official_specs
            assert 'important_parameter' in df.columns
            assert all(df['important_parameter'] == 10)

        else:
            assert df.columns.to_list() == official_specs


@pytest.mark.parametrize('n_paragraphs', [0, 1, 5])
@pytest.mark.parametrize('debug', [True, False], ids=['debug', 'official_spec'])
def test_without_relation(model_entities, debug, n_paragraphs):
    text = 'This is a filler sentence. Britney Spears had a concert in Brazil yesterday. And I am a filler too.'

    models_relations = {}
    texts = n_paragraphs * [(text, {'important_parameter': 10})]
    df = run_pipeline(texts, model_entities, models_relations, debug)

    official_specs = ['entity',
                      'entity_type',
                      'property',
                      'property_value',
                      'property_type',
                      'property_value_type',
                      'ontology_source',
                      'paper_id',
                      'start_char',
                      'end_char']

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_paragraphs * 3  # 3 entities

    if n_paragraphs > 0:
        if debug:
            assert df.columns.to_list() != official_specs
            assert 'important_parameter' in df.columns
            assert all(df['important_parameter'] == 10)

        else:
            assert df.columns.to_list() == official_specs

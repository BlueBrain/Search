"""Collection of tests focused on the bbsearch.mining.pipeline module."""

import pandas as pd
import pytest
from spacy.tokens import Doc, Span

from bbsearch.mining import StartWithTheSameLetter, TextMiningPipeline


def test_overall(model_entities):
    text = 'This is a filler sentence. Britney Spears had a concert in Brazil yesterday. And I am a filler too.'

    # wrong arguments
    with pytest.raises(TypeError):
        pipeline = TextMiningPipeline(model_entities, {('etype_1', 'etype_2'): ['WRONG TYPE']})

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
    pipeline = TextMiningPipeline(model_entities, models_relations)
    df = pipeline(text)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3 + 1 + 1  # 3 entities, 1 ('PERSON', 'DATE') relation and ('PERSON', 'GPE') relation

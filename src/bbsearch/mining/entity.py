"""Named Entity Recognition."""
import pandas as pd


def find_entities(doc, model_entities, return_prob=False, threshold=0.5):
    """Find entities in a given text.

    Parameters
    ----------
    doc : spacy.Doc
        Spacy parsed document, which can be obtained by calling `doc = nlp(raw_text, disable=['ner'])`.
    model_entities : spacy.language.Language
        Spacy model with pipes for parsing and ner, e.g. `model_entities=spacy.load("en_ner_craft_md")`.
    return_prob : bool, optional
        If `True`, the column `confidence_score` of the output table is filled with estimates of the confidence of the
        extracted entities, i.e. float values between 0 and 1.
        Notice that setting `return_prob=True` may return different entities, as to access the confindence scores of
        spaCy it is necessary to perform a forward pass using an undocumented "beam" approach.
    threshold : float, optional
        If `return_prob` is `True`, only extracted entities with `confidence > threshold` are returned.

    Returns
    -------
    table_extractions : pd.DataFrame
        Table containing the extracted entities.

    References
    ----------
    [1] https://allenai.github.io/scispacy/
    [2] https://spacy.io/api/doc
    [3] https://github.com/explosion/spaCy/issues/2601
    """
    headers = ['entity',
               'entity_type',
               'relation_model',
               'property',
               'property_value',
               'property_type',
               'property_value_type',
               'ontology_source',
               'paper_id',
               'start_char',
               'end_char',
               'confidence_score']

    ner = model_entities.get_pipe('ner')

    lines = []
    if not return_prob:
        extracted_ents = ner(doc).ents
        for e in extracted_ents:
            lines.append({
                'entity': e.text,
                'entity_type': e.label_,
                'start_char': e.start_char,
                'end_char': e.end_char
            })
        table_extractions = pd.DataFrame(data=lines, columns=headers)

    else:
        # This is a undocumented spaCy hack to get confidence scores
        ner = model_entities.get_pipe('ner')
        docs = [model_entities(doc.text)]
        beam = ner.beam_parse(docs, beam_width=16)[0]
        entities = ner.moves.get_beam_annot(beam)
        for k, v in entities.items():
            ent_span = doc[k[0]:k[1]]
            lines.append({'entity': ent_span.text,
                          'start_char': ent_span.start_char,
                          'end_char': ent_span.end_char,
                          'entity_type': ner.vocab.strings[k[2]],
                          'confidence_score': v})
        table_extractions = pd.DataFrame(lines, columns=headers)
        table_extractions = table_extractions.loc[table_extractions.confidence_score > threshold]

    return table_extractions

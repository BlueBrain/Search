"""Entire end-to-end pipeline extracting entities and relations from text."""
from .entity import find_entities
from .relation import REModel, annotate

import pandas as pd


def run_pipeline(text, model_entities, models_relations, return_prob=False):
    """Run end-to-end extractions.

    Parameters
    ----------
    text : str
        Arbitrarily long text without any preprocessing.
    model_entities : spacy.language.Language
        Spacy model for entity extraction, e.g. `model_entities=spacy.load("en_ner_craft_md")`.
    models_relations : dict
        The keys are pairs (two element tuples) of entity types (i.e. ('GGP', 'CHEBI')). The first entity type
        is the subject and the second one is the object. Note that the entity types should correspond to those inside
        of `model_entities`. The value is a list of instances of relation extraction models,
        that is instances of some subclass of ``REModel``.
    return_prob : bool, optional
        If True, the output table contains also a column with confidence scores.

    Returns
    -------
    table_extractions : pd.DataFrame
        The final table with all extracted entities, relations, and attributes.
    """
    # sanity checks
    if not all([isinstance(model, REModel) for model_list in models_relations.values() for model in model_list]):
        raise TypeError('Each relation extraction model needs to be a subclass of REModel.')

    headers = ['entity',
               'entity_type',
               'relation_model',
               'property',
               'property_value',
               'property_type',
               'property_value_type',
               'ontology_source',
               'paper_id',
               'start_pos',
               'end_pos']
    if return_prob:
        headers.append('confidence')

    doc = find_entities(text, model_entities)
    lines = []

    for sent in doc.sents:
        detected_entities = [ent for ent in sent.ents]

        for s_ent in detected_entities:
            # add single lines for entities
            lines.append({'entity': s_ent.text,
                          'entity_type': s_ent.label_,
                          'start_pos': s_ent.start_char,
                          'end_pos': s_ent.end_char,
                          })

            # extract relations
            for o_ent in detected_entities:
                if s_ent == o_ent:
                    continue

                so = (s_ent.label_, o_ent.label_)
                if so in models_relations:
                    for re_model in models_relations[so]:
                        annotated_sent = annotate(doc, sent, s_ent, o_ent, re_model.symbols)
                        line_dict = {'entity': s_ent.text,
                                     'entity_type': s_ent.label_,
                                     'relation_model': re_model.__class__.__name__,
                                     'start_pos': s_ent.start_char,
                                     'end_pos': s_ent.end_char,
                                     'property_type': 'relation',
                                     'property_value': o_ent.text,
                                     'property_value_type': o_ent.label_
                                     }
                        if return_prob:
                            line_dict.update(dict(zip(['property', 'confidence'],
                                             re_model.predict(annotated_sent, return_prob=True))))
                        else:
                            line_dict.update({'property': re_model.predict(annotated_sent, return_prob=False)})
                        lines.append(line_dict)

    result = pd.DataFrame(lines, columns=headers)

    return result

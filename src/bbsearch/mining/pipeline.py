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
        Spacy model with pipes for parsing and ner, e.g. `model_entities=spacy.load("en_ner_craft_md")`.
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
                Table containing the extracted entities, relations, and attributes.
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
               'start_char',
               'end_char']
    if return_prob:
        headers.append('confidence')

    doc = model_entities(text, disable=['ner'])

    df_entities = find_entities(doc, model_entities, return_prob)
    rows_relations = []

    for sent in doc.sents:
        # Extract entities
        df_entities_sent = df_entities.loc[(df_entities.start_char >= sent.start_char) &
                                           (df_entities.end_char <= sent.end_char)]
        # Extract relations
        for s_idx, s_ent in df_entities_sent.iterrows():  # potential subject
            for o_idx, o_ent in df_entities_sent.iterrows():  # potential object
                if s_idx == o_idx:  # relations cannot be between an entity and itself
                    continue
                so = (s_ent.entity_type, o_ent.entity_type)
                if so in models_relations:
                    for re_model in models_relations[so]:
                        annotated_sent = annotate(doc, sent, s_ent, o_ent, re_model.symbols)
                        row_relation = {'entity': s_ent.entity,
                                        'entity_type': s_ent.entity_type,
                                        'relation_model': re_model.__class__.__name__,
                                        'start_char': sent.start_char,
                                        'end_char': sent.end_char,
                                        'property_type': 'relation',
                                        'property_value': o_ent.entity,
                                        'property_value_type': o_ent.entity_type
                                        }
                        if return_prob:
                            row_relation.update(dict(zip(['property', 'confidence'],
                                                         re_model.predict(annotated_sent, return_prob=True))))
                        else:
                            row_relation.update({'property': re_model.predict(annotated_sent, return_prob=False)})
                        rows_relations.append(row_relation)

    df_relations = pd.DataFrame(rows_relations, columns=headers)

    return df_entities.append(df_relations, ignore_index=True)

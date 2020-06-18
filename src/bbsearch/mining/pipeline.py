"""Entire end-to-end pipeline extracting entities and relations from text."""

import pandas as pd
import spacy

from .relation import REModel, annotate


def run_pipeline(texts, model_entities, models_relations, debug=False):
    """Run end-to-end extractions.

    Parameters
    ----------
    texts : list
        It is a list of tuples where the first element is some
        text (``str``) and the second element is a dictionary with all relevant metadata.
    model_entities : spacy.lang.en.English
        Spacy model. Note that this model defines entity types.
    models_relations : dict
        The keys are pairs (two element tuples) of entity types (i.e. ('GGP', 'CHEBI')). The first entity type
        is the subject and the second one is the object. Note that the entity types should correspond to those inside
        of `model_entities`. The value is a list of instances of relation extraction models,
        that is instances of some subclass of ``REModel``.

    Returns
    -------
    result : pd.DataFrame
        The final table. If `debug=True` then it contains all the metadata. If False then it
        only contains columns in the official specification.
    """
    # sanity checks
    if not isinstance(model_entities, spacy.language.Language):
        raise TypeError('Current implementation requires `model_entities` to be an instance of '
                        '`spacy.language.Language`. Try with `model_entities==spacy.load("en_ner_craft_md")`')

    if not all([isinstance(model, REModel) for model_list in models_relations.values() for model in model_list]):
        raise TypeError('Each relation extraction model needs to be a subclass of REModel.')

    specs = ['entity',
             'entity_type',
             'property',
             'property_value',
             'property_type',
             'property_value_type',
             'ontology_source',
             'paper_id',
             'start_pos',
             'end_pos']

    docs_gen = model_entities.pipe(texts, disable=['tagger'], as_tuples=True)
    lines = []

    for doc, metadata in docs_gen:
        for sent in doc.sents:
            detected_entities = [ent for ent in sent.ents]

            for s_ent in detected_entities:
                # add single lines for entities
                lines.append(dict(entity=s_ent.text,
                                  entity_type=s_ent.label_,
                                  start_pos=s_ent.start_char,
                                  end_pos=s_ent.end_char,
                                  **metadata)
                             )

                # extract relations
                for o_ent in detected_entities:
                    if s_ent == o_ent:
                        continue

                    so = (s_ent.label_, o_ent.label_)
                    if so in models_relations:
                        for re_model in models_relations[so]:
                            annotated_sent = annotate(doc, sent, s_ent, o_ent, re_model.symbols)
                            property_ = re_model.predict(annotated_sent)
                            lines.append({'entity': s_ent.text,
                                          'entity_type': s_ent.label_,
                                          'relation_model': re_model.__class__.__name__,
                                          'start_pos': s_ent.start_char,
                                          'end_pos': s_ent.end_char,
                                          'property_type': 'relation',
                                          'property': property_,
                                          'property_value': o_ent.text,
                                          'property_value_type': o_ent.label_
                                          })

    df_all = pd.DataFrame(lines)

    if debug:
        return df_all
    else:
        return pd.DataFrame(df_all, columns=specs)

"""Entire end-to-end pipeline extracting entities and relations from text."""
from .entity import find_entities
from .relation import REModel, annotate

import pandas as pd
import spacy


class TextMiningPipeline:
    def __init__(self, model_entities, models_relations):
        """Pipeline for extracting entities, relations, and attributes of interest.

        Parameters
        ----------
        model_entities : spacy.language.Language
            Spacy model with pipes for parsing and ner, e.g. `model_entities=spacy.load("en_ner_craft_md")`.
        models_relations : dict
            The keys are pairs (two element tuples) of entity types (i.e. ('GGP', 'CHEBI')). The first entity type
            is the subject and the second one is the object. Note that the entity types should correspond to those
            inside of `model_entities`. The value is a list of instances of relation extraction models, that is
            instances of some subclass of `REModel`.
        """
        if not isinstance(model_entities, spacy.language.Language):
            raise TypeError('Current implementation requires `model_entities` to be an instance of '
                            '`spacy.language.Language`. Try with `model_entities==spacy.load("en_ner_craft_md")`')
        if not all([isinstance(model, REModel) for model_list in models_relations.values() for model in model_list]):
            raise TypeError('Each relation extraction model needs to be a subclass of REModel.')

        self.model_entities = model_entities
        self.models_relations = models_relations

    def __call__(self, text, return_prob=False, debug=False):
        """Apply pipeline to a given text.

        Parameters
        ----------
        text : str
            Arbitrarily long text without any preprocessing.
        return_prob : bool, optional
            If `True`, the column `confidence_score` of the output table is filled with estimates of the confidence of
            the extracted entities and properties, i.e. float values between 0 and 1.
            Notice that setting `return_prob=True` may return different entities, as to access the confindence scores of
            spaCy it is necessary to perform a forward pass using an undocumented "beam" approach.
        debug : bool, optional
            If `True`, the output table contains extra columns that can be useful to debug the underlying machine
            learning models.

        Returns
        -------
        table_extractions : pd.DataFrame
            Table containing the extracted entities, relations, and attributes.
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

        doc = self.model_entities(text, disable=['ner'])

        # Extract entities in text
        df_entities = find_entities(doc, self.model_entities, return_prob)
        rows_relations = []

        for sent in doc.sents:
            # Select extracted entities in this sentence
            df_entities_sent = df_entities.loc[(df_entities.start_char >= sent.start_char) &
                                               (df_entities.end_char <= sent.end_char)]
            # Extract relations in this sentence
            for s_idx, s_ent in df_entities_sent.iterrows():  # potential subject
                for o_idx, o_ent in df_entities_sent.iterrows():  # potential object
                    if s_idx == o_idx:  # relations cannot be between an entity and itself
                        continue
                    so = (s_ent.entity_type, o_ent.entity_type)
                    if so in self.models_relations:
                        for re_model in self.models_relations[so]:
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
                                row_relation.update(dict(zip(['property', 'confidence_score'],
                                                             re_model.predict(annotated_sent, return_prob=True))))
                            else:
                                row_relation.update({'property': re_model.predict(annotated_sent, return_prob=False)})
                            rows_relations.append(row_relation)

        df_relations = pd.DataFrame(rows_relations, columns=headers)

        table_extractions = df_entities.append(df_relations, ignore_index=True)
        if not debug:
            table_extractions.drop('relation_model', axis=1, inplace=True)

        return table_extractions

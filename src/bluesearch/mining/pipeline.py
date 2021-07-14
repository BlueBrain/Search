"""Complete pipeline to mine entities, relations, attributes from text."""

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

import pandas as pd
import spacy

from .relation import REModel, annotate

SPECS = [
    "entity",
    "entity_type",
    "property",
    "property_value",
    "property_type",
    "property_value_type",
    "ontology_source",
    "paper_id",  # article_id:section_name:paragraph_id
    "start_char",
    "end_char",
]


def run_pipeline(
    texts, model_entities, models_relations, debug=False, excluded_entity_type="NaE"
):
    """Run end-to-end extractions.

    Parameters
    ----------
    texts : iterable
        The elements in `texts` are tuples where the first element is the text
        to be processed and the second element is a dictionary with arbitrary
        metadata for the text. Each key in this dictionary will be used to
        construct a new column in the output data frame and the values will
        appear in the corresponding rows.

        Note that if `debug=False` then the output data frame will have
        exactly the columns specified by `SPECS`. That means that some
        columns produced by the entries in metadata might be dropped, and
        some empty columns might be added.
    model_entities : spacy.lang.en.English
        Spacy model. Note that this model defines entity types.
    models_relations : dict
        The keys are pairs (two element tuples) of entity types
        (i.e. ('GGP', 'CHEBI')). The first entity type is the subject
        and the second one is the object. Note that the entity types
        should correspond to those inside of `model_entities`. The value
        is a list of instances of relation extraction models, that is
        instances of some subclass of ``REModel``.
    debug : bool
        If True, columns are not necessarily matching the specification.
        However, they contain debugging information. If False, then
        matching exactly the specification.
    excluded_entity_type : str or None
        If a str, then all entities with type `not_entity_label` will be
        excluded. If None, then no exclusion will be taking place.

    Returns
    -------
    pd.DataFrame
        The final table. If `debug=True` then it contains all the metadata.
        If False then it only contains columns in the official specification.
    """
    # sanity checks
    if not isinstance(model_entities, spacy.language.Language):
        raise TypeError(
            "Current implementation requires `model_entities` to be an instance "
            "of `spacy.language.Language`. Try for example `model_entities = "
            'bluesearch.utils.load_spacy_model("data_and_models/models/ner_er/'
            'model-chemical")`.'
        )

    if not all(
        [
            isinstance(model, REModel)
            for model_list in models_relations.values()
            for model in model_list
        ]
    ):
        raise TypeError(
            "Each relation extraction model needs to be a subclass of REModel."
        )

    if models_relations:
        disable_pipe = (
            []
        )  # parser is needed to split text into sentences, tagger for EntityRuler
    else:
        disable_pipe = ["parser"]

    docs_gen = model_entities.pipe(texts, disable=disable_pipe, as_tuples=True)
    lines = []

    for doc, metadata in docs_gen:
        subtexts = doc.sents if models_relations else [doc]
        for subtext in subtexts:
            detected_entities = [
                ent
                for ent in subtext.ents
                if excluded_entity_type is None or ent.label_ != excluded_entity_type
            ]

            for s_ent in detected_entities:
                # add single lines for entities
                lines.append(
                    dict(
                        entity=s_ent.text,
                        entity_type=s_ent.label_,
                        start_char=s_ent.start_char,
                        end_char=s_ent.end_char,
                        **metadata
                    )
                )

                # extract relations
                for o_ent in detected_entities:
                    if s_ent == o_ent:
                        continue

                    so = (s_ent.label_, o_ent.label_)
                    if so in models_relations:
                        for re_model in models_relations[so]:
                            annotated_sent = annotate(
                                doc, subtext, s_ent, o_ent, re_model.symbols
                            )
                            property_ = re_model.predict(annotated_sent)
                            lines.append(
                                dict(
                                    entity=s_ent.text,
                                    entity_type=s_ent.label_,
                                    relation_model=re_model.__class__.__name__,
                                    start_char=s_ent.start_char,
                                    end_char=s_ent.end_char,
                                    property_type="relation",
                                    property=property_,
                                    property_value=o_ent.text,
                                    property_value_type=o_ent.label_,
                                    **metadata
                                )
                            )

    # enforce columns if there are no extractions or we are in prod mode
    if not lines or not debug:
        return pd.DataFrame(lines, columns=SPECS)
    else:
        return pd.DataFrame(lines)

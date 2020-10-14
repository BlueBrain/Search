"""Module focusing on entity recognition."""
import copy
import json

import spacy


def dump_jsonl(data, path):
    """Save a list of dictionaries to a jsonl.

    Parameters
    ----------
    data : list
        List of dictionaries (json files).

    path : pathlib.Path
        File where to save it.
    """
    with path.open("w") as f:
        for x in data:
            line = json.dumps(x)
            f.write(line + "\n")


def load_jsonl(path):
    """Read jsonl into a list of dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .jsonl file.

    Returns
    -------
    data : list
        List of dictionaries.
    """
    with path.open("r") as f:
        text = f.read()
        data = [json.loads(jline) for jline in text.splitlines()]

    return data


def remap_entity_type(patterns, etype_mapping):
    """Remap entity types in the patterns to a specific model.

    For each entity type in the patterns try to see whether the model supports it
    and if not relabel the entity type to `NaE`.

    Parameters
    ----------
    patterns : list
        List of patterns.

    etype_mapping : dict
        Keys are our entity type names and values are entity type names inside of the spacy model.

        .. code-block:: Python

            {"CHEMICAL": "CHEBI"}


    Returns
    -------
    adjusted_patterns : list
        Patterns that are supposed to be for a specific spacy model.
    """
    adjusted_patterns = copy.deepcopy(patterns)

    for p in adjusted_patterns:
        label = p['label']

        if label in etype_mapping:
            p['label'] = etype_mapping[label]
        else:
            p['label'] = "NaE"

    return adjusted_patterns


def global2model_patterns(patterns, ee_models_library):
    """Convert global patterns to model specific patterns.

    The `patterns` list can look like this
    [
     {"label": "ORG", "pattern": "Apple"},
     {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}
    ]

    Parameters
    ----------
    patterns : list
        List of patterns where the entity type is always referring to the entity type naming
        convention we have internally ("entity_type" column of `ee_models_library`).

    ee_models_library : pd.DataFrame
        3 columns DataFrame connecting model location, our naming and model naming of entity type.
        * "entity_type": our naming of entity_types
        * "model": path to the model folder
        * "entity_type_name": internal name of entities

    Returns
    -------
    res : dict
        The keys are the locations of the model and the values are list of patterns that one
        can load with `EntityRuler(nlp, patterns=patterns)`

    """
    all_models = ee_models_library["model"].unique()
    res = {}
    for model in all_models:
        etype_mapping = {row["entity_type"]: row["entity_type_name"]
                         for _, row in ee_models_library[ee_models_library["model"] == model].iterrows()}
        res[model] = remap_entity_type(patterns, etype_mapping)

    return res


def check_patterns_agree(model, patterns):
    """Validate whether patterns of an existing model agree with given patterns.

    Parameters
    ----------
    model : spacy.Language
        A model that contains an `EntityRuler`.

    patterns : list
        List of patterns.

    Returns
    -------
    res : bool
        If True, the patterns agree.

    Raises
    ------
    ValueError
        The model does not contain an entity ruler or it contains more than 1.
    """
    all_er = [pipe for pipe in model.pipeline if isinstance(pipe[1], spacy.pipeline.entityruler.EntityRuler)]

    if not all_er:
        raise ValueError("The model contains no EntityRuler")
    elif len(all_er) > 1:
        raise ValueError("The model contains more than 1 EntityRuler")
    else:
        return patterns == all_er[0][1].patterns

"""Module focusing on entity recognition."""
import copy
import json


def dump_jsonl(data, path):
    """Save a list of dictionaries to a jsonl.

    Parameters
    ----------
    data : list
        List of dictionaries (json files).

    path : pathlib.Path
        File where to save it il.
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
        ```
        {"CHEMICAL": "CHEBI"}
        ```

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

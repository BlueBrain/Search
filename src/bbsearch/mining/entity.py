"""Module focusing on entity recognition."""
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

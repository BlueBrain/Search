"""Classes and functions for entity extraction (aka named entity recognition)."""

import copy

import numpy as np
import pandas as pd
import spacy

from bbsearch.utils import JSONL


class PatternCreator:
    """Utility class for easy handling of patterns.

    Parameters
    ----------
    storage : None or pd.DataFrame
        If provided, we automatically populate `_storage` with it. If None, then
        we start from scratch - no patterns.

    Attributes
    ----------
    _storage : pd.DataFrame
        A representation of all patterns allows for comfortable sorting,
        filtering, etc. Note that each row represents a single pattern.

    """

    @classmethod
    def load(cls, path):
        """Load from a jsonl file.

        Parameters
        ----------
        path : pathlib.Path
            Path to a jsonl file with patterns.

        Returns
        -------
        pattern_creator : PatternCreator
            Instance of a ``PatternCreator``.
        """
        inst = cls()
        patterns = JSONL.load_jsonl(path)

        for p in patterns:
            inst.add(label=p['label'], pattern=p['pattern'])

        return inst

    def __init__(self, storage=None):
        if storage is None:
            columns = ["label"]
            self._storage = pd.DataFrame(columns=columns)
        else:
            self._storage = storage.reset_index(drop=True)

    def __eq__(self, other):
        """Determine if equal.

        Parameters
        ----------
        other : PatternCreator
            Some other PatternCreator that we wish to compare to.

        Returns
        -------
        bool
            If True, the patterns are identical. Note that the order does not matter.
        """
        if not isinstance(other, self.__class__):
            return False

        self_df_unsorted = self.to_df()
        other_df_unsorted = other.to_df()

        if set(self_df_unsorted.columns) != set(other_df_unsorted.columns):
            return False

        sort_by = list(self_df_unsorted.columns)
        self_df_sorted = self_df_unsorted.sort_values(by=sort_by)
        other_df_sorted = other_df_unsorted.sort_values(by=sort_by)

        self_is_nan = self_df_sorted.isnull().values
        other_is_nan = other_df_sorted.isnull().values

        return np.array_equal(self_is_nan,
                              other_is_nan) and np.array_equal(self_df_sorted.values[~self_is_nan],
                                                               other_df_sorted.values[~other_is_nan])

    def add(self, label, pattern, check_exists=True):
        """Add a single raw in the patterns.

        Parameters
        ----------
        label : str
            Entity type to associate with a given pattern.

        pattern : str or dict or list
            The pattern we want to match. The behavior depends on the type.

            - ``str``: can be used for exact matching (case sensitive). We internally convert
              it to a single-token pattern `{"TEXT": pattern}`.
            - ``dict``: a single-token pattern. This dictionary can contain at most 2 entries.
              The first one represents the attribute: value pair ("LEMMA": "world"). The second
              has a key "OP" and is optional. It represents the operator/quantifier to be used.
              An example of a valid pattern dict is `{"LEMMA": "world", "OP": "+"}`. Note that
              it would detect entities like "world" and "world world world".
            - ``list``: a multi-token pattern. A list of dictionaries that are of the same form
              as described above.

        check_exists : bool
            If True, we only allow to add patterns that do not exist yet.
        """
        if isinstance(pattern, str):
            pattern_ = [{"TEXT": pattern}]

        elif isinstance(pattern, dict):
            pattern_ = [pattern]

        elif isinstance(pattern, list):
            pattern_ = pattern

        else:
            raise TypeError("Unsupported type of pattern")

        new_row = self.raw2row({"label": label, "pattern": pattern_})

        new_storage = self._storage.append(new_row.to_frame().T, ignore_index=True)
        if check_exists and new_storage.duplicated().any():
            raise ValueError("The pattern already exists")

        self._storage = new_storage

    def drop(self, labels):
        """Drop one or multiple patterns.

        Parameters
        ----------
        labels : int or list
            If ``int`` then represent a row index to be dropped. If ``list`` then
            a collection of row indices to be dropped.
        """
        self._storage.drop(index=labels, inplace=True)

    def to_df(self):
        """Convert to DataFrame."""
        return self._storage

    def to_list(self, sort_by=None):
        """Convert to list.

        Parameters
        ----------
        sort_by : None or list
            If None, then no sorting taking place. If ``list``, then the names of columns
            along which to sort.
        """
        sorted_storage = self._storage.sort_values(by=sort_by) if sort_by is not None else self._storage
        return [self.row2raw(row) for _, row in sorted_storage.iterrows()]

    def save(self, path, sort_by=None):
        """Save to jsonl.

        Parameters
        ----------
        sort_by : None or list
            If None, then no sorting taking place. If ``list``, then the names of columns
            along which to sort.
        """
        patterns = self.to_list(sort_by=sort_by)
        JSONL.dump_jsonl(patterns, path)

    def test(self, text, model=None, disable=None, **add_pipe_kwargs):
        """Test the current patterns on text.

        Parameters
        ----------
        text : str
            Some text.

        model : spacy.language.Language or None
            Spacy model. If not provided we default to `spacy.blank("en")`.

        disable : list or None
            List of elements to remove from the pipeline.

        **add_pipe_kwargs : dict
            Additionally parameters to be passed into the `add_pipe` method. Note that
            one can control the position the ``EntityRuler`` this way. If not specified
            we put at at the very end.

        Returns
        -------
        doc : spacy.Doc
            Doc containing the entities under the `ents` property.
        """
        model = model or spacy.blank("en")
        disable = disable or []
        add_pipe_kwargs = add_pipe_kwargs or {"last": True}
        er = spacy.pipeline.EntityRuler(model, patterns=self.to_list(), validate=True)
        model.add_pipe(er, **add_pipe_kwargs)

        return model(text, disable=disable)

    @staticmethod
    def raw2row(raw):
        """Convert an element of patterns list to a pd.Series.

        The goal of this function is to create a pd.Series
        with all entries being strings. This will allow us
        to check for duplicates between different rows really
        quickly.

        Parameters
        ----------
        raw : dict
            Dictionary with two keys: 'label' and 'pattern'.
            The `pattern` needs to be a list of dictionaries
            each representing a pattern for a given token.
            The `label` represents the entity type.

        Returns
        -------
        row : pd.Series
            The index contains the following elements: "label",
            "attribute_0", "value_0", "value_type_0", "op_0",
            "attribute_1", "value_1", "value_type_1", "op_1",
            ...

        """
        d = {"label": raw["label"]}
        for token_ix, e in enumerate(raw["pattern"]):

            if len(e) == 1:
                pass
            elif len(e) == 2 and "OP" in e:
                pass
            else:
                raise ValueError('Invalid element, multi-attribute matches are not supported')

            attribute = list(e)[0]
            value_type = type(e[attribute]).__name__
            value = str(e[attribute])
            op = e.get('OP', "")

            d.update({f"attribute_{token_ix}": attribute,
                      f"value_{token_ix}": value,
                      f"value_type_{token_ix}": value_type,
                      f"op_{token_ix}": op})
        return pd.Series(d)

    @staticmethod
    def row2raw(row):
        """Convert pd.Series to a valid pattern dictionary.

        Note that the `value_{i}` is always a string, however,
        we cast it to `value_type_{i}` type. In most cases the
        type will be ``int``, ``str`` or ``dict``. Since
        this casting is done dynamically we use `eval`.

        Parameters
        ----------
        row : pd.Series
            The index contains the following elements: "label",
            "attribute_0", "value_0", "value_type_0", "op_0",
            "attribute_1", "value_1", "value_type_1", "op_1",

        Returns
        -------
        raw : dict
            Dictionary with two keys: 'label' and 'pattern'.
            The `pattern` needs to be a list of dictionaries
            each representing a pattern for a given token.
            The `label` represents the entity type.
        """
        pattern = []
        token_ix = 0
        while True:
            try:
                attribute = row[f"attribute_{token_ix}"]  # str
                value_str = row[f"value_{token_ix}"]  # str
                value_type = row[f"value_type_{token_ix}"]

                value = eval(f"{value_type}({value_str})") if value_type != 'str' else value_str
                op = row[f"op_{token_ix}"]  # str

                token_pattern = {attribute: value}
                if op:
                    token_pattern["OP"] = op

                pattern.append(token_pattern)
            except (KeyError, NameError):
                break

            token_ix += 1

        return {"label": row["label"], 'pattern': pattern}


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
        label = p["label"]

        if label in etype_mapping:
            p["label"] = etype_mapping[label]
        else:
            p["label"] = "NaE"

    return adjusted_patterns


def global2model_patterns(patterns, ee_models_library):
    """Convert global patterns to model specific patterns.

    The `patterns` list can look like this

    .. code-block:: Python

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
    res = {}
    for model, ee_models_library_slice in ee_models_library.groupby("model"):
        etype_mapping = {
            row["entity_type"]: row["entity_type_name"]
            for _, row in ee_models_library_slice.iterrows()
        }
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
    all_er = [
        pipe
        for _, pipe in model.pipeline
        if isinstance(pipe, spacy.pipeline.EntityRuler)
    ]

    if not all_er:
        raise ValueError("The model contains no EntityRuler")
    elif len(all_er) > 1:
        raise ValueError("The model contains more than 1 EntityRuler")
    else:
        return patterns == all_er.pop().patterns

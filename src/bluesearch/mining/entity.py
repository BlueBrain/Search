"""Classes and functions for entity extraction (aka named entity recognition)."""

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

import ast
import copy

import numpy as np
import pandas as pd
import spacy

from ..utils import JSONL


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

    Examples
    --------
    >>> from bluesearch.mining import PatternCreator
    >>>
    >>> pc = PatternCreator()
    >>> pc.add("FOOD", [{"LOWER": "oreo"}])
    >>> pc.add("DRINK", [{"LOWER": {"REGEX": "^w"}}, {"LOWER": "milk"}])
    >>> doc = pc("It is necessary to dip the oreo in warm milk!")
    >>> [(str(e), e.label_) for e in doc.ents]
    [('oreo', 'FOOD'), ('warm milk', 'DRINK')]
    """

    def __init__(self, storage=None):
        if storage is None:
            columns = ["label"]
            self._storage = pd.DataFrame(columns=columns)
        else:
            self._storage = storage.reset_index(drop=True)

    def __call__(self, text, model=None, disable=None, **add_pipe_kwargs):
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
            we put it at the very end.

        Returns
        -------
        doc : spacy.Doc
            Doc containing the entities under the `ents` property.
        """
        model = model or spacy.blank("en")
        disable = disable or []
        add_pipe_kwargs = add_pipe_kwargs or {"last": True}
        er = model.add_pipe(
            "entity_ruler", config={"validate": True}, **add_pipe_kwargs
        )
        er.add_patterns(self.to_list())

        return model(text, disable=disable)

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

        return np.array_equal(self_is_nan, other_is_nan) and np.array_equal(
            self_df_sorted.values[~self_is_nan], other_df_sorted.values[~other_is_nan]
        )

    def add(self, label, pattern, check_exists=True):
        """Add a single raw in the patterns.

        Parameters
        ----------
        label : str
            Entity type to associate with a given pattern.
        pattern : str or dict or list
            The pattern we want to match. The behavior depends on the type.

            - ``str``: can be used for exact matching (case sensitive). We
              internally convert it to a single-token pattern `{"TEXT": pattern}`.
            - ``dict``: a single-token pattern. This dictionary can contain
              at most 2 entries. The first one represents the attribute:
              value pair ("LEMMA": "world"). The second has a key "OP" and is
              optional. It represents the operator/quantifier to be used.
              An example of a valid pattern dict is
              `{"LEMMA": "world", "OP": "+"}`. Note that it would detect
              entities like "world" and "world world world".
            - ``list``: a multi-token pattern. A list of dictionaries that
              are of the same form as described above.

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
        self._storage = self._storage.drop(index=labels).reset_index(drop=True)

    def to_df(self):
        """Convert to a pd.DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the `_storage`. Each row represents a single entity type
            pattern. All elements are strings.
        """
        return self._storage.copy()

    def to_list(self, sort_by=None):
        """Convert to a list.

        Parameters
        ----------
        sort_by : None or list
            If None, then no sorting taking place. If ``list``, then the
            names of columns along which to sort.

        Returns
        -------
        list
            A list where each element represents one entity type pattern.
            Note that this list can be directly passed into the `EntityRuler`.
        """
        storage = self.to_df()
        sorted_storage = (
            storage.sort_values(by=sort_by) if sort_by is not None else storage
        )
        return [self.row2raw(row) for _, row in sorted_storage.iterrows()]

    def to_jsonl(self, path, sort_by=None):
        """Save to JSONL.

        Parameters
        ----------
        path : pathlib.Path
            File where to save it.
        sort_by : None or list
            If None, then no sorting taking place. If ``list``, then the
            names of columns along which to sort.
        """
        patterns = self.to_list(sort_by=sort_by)
        JSONL.dump_jsonl(patterns, path)

    @classmethod
    def from_jsonl(cls, path):
        """Load from a JSONL file.

        Parameters
        ----------
        path : pathlib.Path
            Path to a JSONL file with patterns.

        Returns
        -------
        pattern_creator : bluesearch.mining.PatternCreator
            Instance of a ``PatternCreator``.
        """
        inst = cls()
        patterns = JSONL.load_jsonl(path)

        for p in patterns:
            inst.add(label=p["label"], pattern=p["pattern"])

        return inst

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
            Dictionary with two keys: "label" and "pattern".
            The `pattern` needs to be a list of dictionaries
            each representing a pattern for a given token.
            The `label` is a string representing the entity type.

        Returns
        -------
        row : pd.Series
            The index contains the following elements: "label",
            "attribute_0", "value_0", "value_type_0", "op_0",
            "attribute_1", "value_1", "value_type_1", "op_1",
            ...
        """
        if not isinstance(raw["label"], str):
            raise TypeError("The label needs to be a string")

        if not isinstance(raw["pattern"], list):
            raise TypeError("The pattern needs to be a list")

        d = {"label": raw["label"]}
        for token_ix, e in enumerate(raw["pattern"]):
            if not isinstance(e, dict):
                raise TypeError("The per token pattern needs to be a dictionary")

            if len(e) == 1:
                pass
            elif len(e) == 2 and "OP" in e:
                pass
            else:
                raise ValueError(
                    "Invalid element, multi-attribute matches are not supported"
                )

            attribute = next(filter(lambda key: key != "OP", e))
            value_type = type(e[attribute]).__name__
            value = str(e[attribute])
            op = e.get("OP", "")

            d.update(
                {
                    f"attribute_{token_ix}": attribute,
                    f"value_{token_ix}": value,
                    f"value_type_{token_ix}": value_type,
                    f"op_{token_ix}": op,
                }
            )
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
            Dictionary with two keys: "label" and "pattern".
            The `pattern` needs to be a list of dictionaries
            each representing a pattern for a given token.
            The `label` is a string representing the entity type.
        """
        pattern = []
        token_ix = 0
        while True:
            try:
                attribute = row[f"attribute_{token_ix}"]  # str
                value_str = row[f"value_{token_ix}"]  # str
                value_type = row[f"value_type_{token_ix}"]  # str
                op = row[f"op_{token_ix}"]  # str

                if any(
                    not isinstance(x, str)
                    for x in [attribute, value_str, value_type, op]
                ):
                    raise KeyError()

                if value_type != "str":
                    try:
                        value = ast.literal_eval(value_str)
                    except ValueError as ve:
                        if str(ve).startswith("malformed node or string"):
                            raise NameError(str(ve)) from ve
                        else:
                            raise
                else:
                    value = value_str

                token_pattern = {attribute: value}
                if op:
                    token_pattern["OP"] = op

                pattern.append(token_pattern)
            except KeyError:
                break

            token_ix += 1

        if token_ix == 0:
            raise ValueError("No valid pattern was found")

        return {"label": row["label"], "pattern": pattern}


def global2model_patterns(patterns, entity_type):
    """Remap entity types in the patterns to a specific model.

    For each entity type in the patterns try to see whether the model supports it
    and if not relabel the entity type to `NaE`.

    Parameters
    ----------
    patterns : list
        List of patterns.
    entity_type : str
        Entity type detected by a spacy model.

    Returns
    -------
    adjusted_patterns : list
        Patterns that are supposed to be for a specific spacy model.
    """
    adjusted_patterns = copy.deepcopy(patterns)

    for p in adjusted_patterns:
        label = p["label"]

        if label.lower() != entity_type.lower():
            p["label"] = "NaE"

    return adjusted_patterns


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

"""Subpackage for text mining."""

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

from .attribute import AttributeAnnotationTab, AttributeExtractor, TextCollectionWidget
from .entity import (
    PatternCreator,
    check_patterns_agree,
    global2model_patterns,
    remap_entity_type,
)
from .eval import annotations2df, spacy2df
from .pipeline import SPECS, run_pipeline
from .relation import ChemProt, REModel, StartWithTheSameLetter, annotate

__all__ = [
    "AttributeExtractor",
    "AttributeAnnotationTab",
    "ChemProt",
    "PatternCreator",
    "REModel",
    "SPECS",
    "StartWithTheSameLetter",
    "TextCollectionWidget",
    "annotate",
    "annotations2df",
    "check_patterns_agree",
    "global2model_patterns",
    "remap_entity_type",
    "run_pipeline",
    "spacy2df",
]

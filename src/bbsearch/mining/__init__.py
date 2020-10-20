"""Subpackage for text mining."""

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

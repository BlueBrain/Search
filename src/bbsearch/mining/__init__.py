"""Subpackage for text mining."""

from .attributes import (
    AttributeExtractor,
    AttributeAnnotationTab,
    TextCollectionWidget)
from .entity import find_entities
from .pipeline import TextMiningPipeline
from .relation import ChemProt, REModel, StartWithTheSameLetter, annotate

__all__ = [
    'AttributeExtractor',
    'AttributeAnnotationTab',
    'TextCollectionWidget',
    'ChemProt',
    'REModel',
    'StartWithTheSameLetter',
    'annotate',
    'find_entities',
    'TextMiningPipeline'
]

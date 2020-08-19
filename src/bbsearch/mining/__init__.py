"""Subpackage for text mining."""

from .attributes import AttributeAnnotationTab, AttributeExtractor, TextCollectionWidget
from .eval import prodigy2df, spacy2df
from .pipeline import SPECS, run_pipeline
from .relation import ChemProt, REModel, StartWithTheSameLetter, annotate

__all__ = [
    'AttributeExtractor',
    'AttributeAnnotationTab',
    'TextCollectionWidget',
    'ChemProt',
    'REModel',
    'SPECS',
    'StartWithTheSameLetter',
    'annotate',
    'run_pipeline',
    'prodigy2df',
    'spacy2df'
]

"""Subpackage for text mining."""

from .attributes import AttributeAnnotationTab, AttributeExtractor, TextCollectionWidget
from .entity import dump_jsonl, load_jsonl
from .eval import annotations2df, spacy2df
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
    'annotations2df',
    'dump_jsonl',
    'load_jsonl',
    'run_pipeline',
    'spacy2df'
]

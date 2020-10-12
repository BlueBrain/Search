"""Subpackage for text mining."""

from .attributes import AttributeAnnotationTab, AttributeExtractor, TextCollectionWidget
from .entity import dump_jsonl, global2model_patterns, load_jsonl, remap_entity_type
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
    'global2model_patterns',
    'load_jsonl',
    'remap_entity_type',
    'run_pipeline',
    'spacy2df'
]

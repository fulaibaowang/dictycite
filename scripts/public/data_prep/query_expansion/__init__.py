"""YAML-driven query expansion from tabular entity files (JSONL enrichment)."""

from .config import ExpansionConfig, load_expansion_config
from .expand import expand_query_structured
from .table import EntityIndex, build_entity_index

__all__ = [
    "ExpansionConfig",
    "load_expansion_config",
    "EntityIndex",
    "build_entity_index",
    "expand_query_structured",
]

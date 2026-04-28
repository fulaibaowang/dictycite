from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TableConfig:
    separator: str = "\t"


@dataclass
class QueryConfig:
    read_field: str = "query_text"


@dataclass
class FiltersConfig:
    min_alias_len: int = 4
    max_aliases_per_entity: int = 3
    auto_block_if_seen_in_n_entities: int = 5
    blocklist: List[str] = field(default_factory=list)


@dataclass
class ExpandVariantSpec:
    output_field: str
    expand_with: List[str]
    exclude_from_output: List[str] = field(default_factory=list)
    structured_strict: Optional[bool] = None


@dataclass
class ExpansionConfig:
    entity_id_col: str
    detect_from: List[str]
    expand_variants: Dict[str, ExpandVariantSpec]
    filters: FiltersConfig
    table: TableConfig = field(default_factory=TableConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    comma_split_cols: List[str] = field(default_factory=list)
    product_cols: List[str] = field(default_factory=list)
    entity_id_pattern: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExpansionConfig":
        if "entity_id_col" not in raw:
            raise ValueError("config: missing entity_id_col")
        if "detect_from" not in raw or not raw["detect_from"]:
            raise ValueError("config: missing detect_from")
        if "expand_variants" not in raw or not raw["expand_variants"]:
            raise ValueError("config: missing expand_variants")

        fv: Dict[str, ExpandVariantSpec] = {}
        for name, spec in raw["expand_variants"].items():
            if not isinstance(spec, dict):
                raise ValueError(f"expand_variants.{name}: expected mapping")
            ew = spec.get("expand_with")
            if not ew or not isinstance(ew, list):
                raise ValueError(f"expand_variants.{name}: expand_with (list) required")
            out_field = spec.get("output_field")
            if not out_field:
                raise ValueError(f"expand_variants.{name}: output_field required")
            fv[name] = ExpandVariantSpec(
                output_field=str(out_field),
                expand_with=list(ew),
                exclude_from_output=list(spec.get("exclude_from_output") or []),
                structured_strict=spec.get("structured_strict"),
            )

        f_raw = raw.get("filters") or {}
        filters = FiltersConfig(
            min_alias_len=int(f_raw.get("min_alias_len", 4)),
            max_aliases_per_entity=int(f_raw.get("max_aliases_per_entity", 3)),
            auto_block_if_seen_in_n_entities=int(
                f_raw.get("auto_block_if_seen_in_n_entities", 5)
            ),
            blocklist=[str(x).lower().strip() for x in (f_raw.get("blocklist") or []) if str(x).strip()],
        )

        t_raw = raw.get("table") or {}
        q_raw = raw.get("query") or {}

        return cls(
            entity_id_col=str(raw["entity_id_col"]),
            detect_from=list(raw["detect_from"]),
            expand_variants=fv,
            filters=filters,
            table=TableConfig(separator=str(t_raw.get("separator", "\t"))),
            query=QueryConfig(read_field=str(q_raw.get("read_field", "query_text"))),
            comma_split_cols=[str(x) for x in (raw.get("comma_split_cols") or [])],
            product_cols=[str(x) for x in (raw.get("product_cols") or [])],
            entity_id_pattern=str(raw.get("entity_id_pattern") or ""),
        )

    def validate(self) -> None:
        for vk, spec in self.expand_variants.items():
            if not spec.output_field:
                raise ValueError(f"expand_variants.{vk}: output_field must be non-empty")
        if self.entity_id_pattern:
            try:
                re.compile(self.entity_id_pattern)
            except re.error as e:
                raise ValueError(f"entity_id_pattern is not a valid regex: {e}") from e


def load_expansion_config(path: Path) -> ExpansionConfig:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path}: expected YAML mapping at top level")
    cfg = ExpansionConfig.from_dict(raw)
    cfg.validate()
    return cfg

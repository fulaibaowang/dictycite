from __future__ import annotations

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
    expand_with: List[str]
    exclude_output: List[str] = field(default_factory=list)
    structured_strict: Optional[bool] = None


@dataclass
class ExpansionConfig:
    entity_id_col: str
    detect_from: List[str]
    expand_variants: Dict[str, ExpandVariantSpec]
    filters: FiltersConfig
    table: TableConfig = field(default_factory=TableConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    expansion_outputs: Dict[str, str] = field(default_factory=dict)
    comma_split_columns: List[str] = field(default_factory=list)
    product_columns: List[str] = field(default_factory=list)
    ddb_id_pattern: str = r"\bDDB_G\d+\b"
    max_genes_total: int = 5
    structured_max_aliases_per_gene: int = 4

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
            fv[name] = ExpandVariantSpec(
                expand_with=list(ew),
                exclude_output=list(spec.get("exclude_output") or []),
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
        eo = raw.get("expansion_outputs") or {}

        return cls(
            entity_id_col=str(raw["entity_id_col"]),
            detect_from=list(raw["detect_from"]),
            expand_variants=fv,
            filters=filters,
            table=TableConfig(separator=str(t_raw.get("separator", "\t"))),
            query=QueryConfig(read_field=str(q_raw.get("read_field", "query_text"))),
            expansion_outputs={str(k): str(v) for k, v in eo.items()},
            comma_split_columns=[str(x) for x in (raw.get("comma_split_columns") or [])],
            product_columns=[str(x) for x in (raw.get("product_columns") or [])],
            ddb_id_pattern=str(raw.get("ddb_id_pattern", r"\bDDB_G\d+\b")),
            max_genes_total=int(raw.get("max_genes_total", 5)),
            structured_max_aliases_per_gene=int(raw.get("structured_max_aliases_per_gene", 4)),
        )

    def validate(self) -> None:
        for vk, out_key in self.expansion_outputs.items():
            if vk not in self.expand_variants:
                raise ValueError(
                    f"expansion_outputs key {vk!r} not found in expand_variants"
                )
            if not out_key:
                raise ValueError(f"expansion_outputs.{vk}: empty output field name")
        for vk in self.expand_variants:
            if vk not in self.expansion_outputs:
                raise ValueError(
                    f"expand_variants.{vk!r} missing from expansion_outputs "
                    "(each variant needs a JSON output key)"
                )


def load_expansion_config(path: Path) -> ExpansionConfig:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path}: expected YAML mapping at top level")
    cfg = ExpansionConfig.from_dict(raw)
    cfg.validate()
    return cfg

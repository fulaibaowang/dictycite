from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl

from .config import ExpansionConfig, ExpandVariantSpec


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _split_syn(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.upper() == "NA":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _cell_tokens(val: Any, col: str, comma_split_cols: Set[str]) -> List[str]:
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.upper() == "NA":
        return []
    if col in comma_split_cols:
        return _split_syn(val)
    return [s]


def _is_good_alias(
    a: str,
    *,
    blocklist: Set[str],
    min_len: int,
    alias_gene_count: Dict[str, int],
    freq_cap: int,
) -> bool:
    if not a:
        return False
    s = a.strip()
    sl = s.lower()
    if not s:
        return False
    if sl in blocklist:
        return False
    if len(sl) < min_len:
        return False
    if alias_gene_count.get(sl, 0) >= freq_cap:
        return False
    return True


def _dedupe_preserve_order(tokens: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


@dataclass
class EntityIndex:
    cfg: ExpansionConfig
    alias_to_gene: Dict[str, str]
    ambig_alias: Set[str]
    token_re: re.Pattern
    variant_gene_to_aliases: Dict[str, Dict[str, List[str]]]
    variant_include_product: Dict[str, bool]
    variant_structured_strict: Dict[str, bool]
    gene_to_product: Dict[str, Optional[str]]

    def variant_keys_in_order(self) -> List[str]:
        return list(self.cfg.expand_variants.keys())


def _primary_gene_name_token(
    row: Dict[str, Any],
    spec: ExpandVariantSpec,
    comma_split: Set[str],
    detect_from: Set[str],
) -> Optional[str]:
    """First detection-eligible expand_with column: first token (matches notebook gname_out)."""
    exclude = set(spec.exclude_from_output)
    for col in spec.expand_with:
        if col not in detect_from or col in exclude:
            continue
        if col not in row:
            continue
        toks = _cell_tokens(row.get(col), col, comma_split)
        for t in toks:
            ts = t.strip()
            if not ts or ts.upper() == "NA":
                continue
            return ts
    return None


def _build_expansion_list_for_row(
    row: Dict[str, Any],
    spec: ExpandVariantSpec,
    comma_split: Set[str],
    blocklist: Set[str],
    alias_gene_count: Dict[str, int],
    detect_from: Set[str],
    min_alias_len: int,
    max_aliases: int,
    freq_cap: int,
) -> List[str]:
    """
    Ordered expansion aliases per entity, capped.

    Only columns that are in detect_from contribute alias tokens (comma-split,
    filtered). Columns in expand_with but not detect_from are expansion-only
    (whole string, handled separately as product strings).

    Like notebooks/query_expansion_bm25.py: the primary gene name is always kept
    first (no min-length / frequency filters); remaining tokens are filtered.
    """
    exclude = set(spec.exclude_from_output)
    primary = _primary_gene_name_token(row, spec, comma_split, detect_from)

    raw: List[str] = []
    for col in spec.expand_with:
        if col not in detect_from:
            continue
        if col in exclude:
            continue
        if col not in row:
            continue
        raw.extend(_cell_tokens(row.get(col), col, comma_split))
    raw = _dedupe_preserve_order(raw)

    out: List[str] = []
    if primary:
        out.append(primary)

    for a in raw:
        al = a.lower()
        if primary and al == primary.lower():
            continue
        if not _is_good_alias(
            a,
            blocklist=blocklist,
            min_len=min_alias_len,
            alias_gene_count=alias_gene_count,
            freq_cap=freq_cap,
        ):
            continue
        out.append(a)
        if len(out) >= max_aliases:
            break
    return out


def _product_for_row(row: Dict[str, Any], product_like_cols: List[str]) -> Optional[str]:
    """Return concatenated values of expansion-only columns (not in detect_from)."""
    parts: List[str] = []
    for col in product_like_cols:
        if col not in row:
            continue
        v = row.get(col)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.upper() != "NA":
            parts.append(s)
    return ", ".join(parts) if parts else None


def build_entity_index(table_path: Path, cfg: ExpansionConfig) -> EntityIndex:
    sep = cfg.table.separator
    df = pl.read_csv(
        table_path,
        separator=sep,
        infer_schema_length=10_000,
        ignore_errors=True,
    )
    cols = df.columns
    for c in [cfg.entity_id_col, *cfg.detect_from]:
        if c not in cols:
            raise ValueError(f"table {table_path}: missing column {c!r}")
    for spec in cfg.expand_variants.values():
        for c in spec.expand_with:
            if c not in cols:
                raise ValueError(f"table {table_path}: missing column {c!r} (expand_with)")

    comma_split = set(cfg.comma_split_cols)
    detect_from = set(cfg.detect_from)
    blocklist = set(cfg.filters.blocklist)
    min_alias_len = cfg.filters.min_alias_len
    max_aliases = cfg.filters.max_aliases_per_entity
    freq_cap = cfg.filters.auto_block_if_seen_in_n_entities
    rows = df.to_dicts()

    # Columns in any variant's expand_with that are NOT in detect_from are
    # expansion-only (whole-string, not alias-filtered, not used for detection).
    all_product_like: List[str] = list(dict.fromkeys(
        col
        for spec in cfg.expand_variants.values()
        for col in spec.expand_with
        if col not in detect_from
    ))

    # Pass 1: alias -> number of distinct entities (frequency), same as notebook
    alias_gene_count: Dict[str, int] = {}
    for row in rows:
        gid = row.get(cfg.entity_id_col)
        if gid is None:
            continue
        gid = str(gid).strip()
        if not gid or gid.upper() == "NA":
            continue
        seen_raw: Set[str] = set()
        for col in cfg.detect_from:
            for a in _cell_tokens(row.get(col), col, comma_split):
                if not a:
                    continue
                al = a.lower()
                if al not in seen_raw:
                    seen_raw.add(al)
                    alias_gene_count[al] = alias_gene_count.get(al, 0) + 1

    variant_gene_to_aliases: Dict[str, Dict[str, List[str]]] = {}
    variant_include_product: Dict[str, bool] = {}
    variant_structured_strict: Dict[str, bool] = {}

    for vk, spec in cfg.expand_variants.items():
        gmap: Dict[str, List[str]] = {}
        for row in rows:
            gid = row.get(cfg.entity_id_col)
            if gid is None:
                continue
            gid = str(gid).strip()
            if not gid or gid.upper() == "NA":
                continue
            gid_u = gid.upper()
            gmap[gid_u] = _build_expansion_list_for_row(
                row, spec, comma_split, blocklist, alias_gene_count,
                detect_from, min_alias_len, max_aliases, freq_cap,
            )
        variant_gene_to_aliases[vk] = gmap
        include_prod = any(c not in detect_from for c in spec.expand_with)
        variant_include_product[vk] = include_prod
        if spec.structured_strict is not None:
            variant_structured_strict[vk] = spec.structured_strict
        else:
            variant_structured_strict[vk] = not include_prod

    gene_to_product: Dict[str, Optional[str]] = {}
    for row in rows:
        gid = row.get(cfg.entity_id_col)
        if gid is None:
            continue
        gid = str(gid).strip()
        if not gid or gid.upper() == "NA":
            continue
        gene_to_product[gid.upper()] = _product_for_row(row, all_product_like)

    # alias_to_gene: register tokens from union of all variants' expansion lists
    alias_to_gene: Dict[str, str] = {}
    ambig_alias: Set[str] = set()

    for gid_u in gene_to_product:
        seen_tokens: Set[str] = set()
        for vk in cfg.expand_variants:
            for a in variant_gene_to_aliases[vk].get(gid_u, []):
                a_norm = a.lower()
                if a_norm in seen_tokens:
                    continue
                seen_tokens.add(a_norm)
                if not _is_good_alias(
                    a,
                    blocklist=blocklist,
                    min_len=cfg.filters.min_alias_len,
                    alias_gene_count=alias_gene_count,
                    freq_cap=cfg.filters.auto_block_if_seen_in_n_entities,
                ):
                    continue
                if a_norm in ambig_alias:
                    continue
                if a_norm in alias_to_gene and alias_to_gene[a_norm] != gid_u:
                    ambig_alias.add(a_norm)
                    alias_to_gene.pop(a_norm, None)
                else:
                    alias_to_gene[a_norm] = gid_u

    return EntityIndex(
        cfg=cfg,
        alias_to_gene=alias_to_gene,
        ambig_alias=ambig_alias,
        token_re=TOKEN_RE,
        variant_gene_to_aliases=variant_gene_to_aliases,
        variant_include_product=variant_include_product,
        variant_structured_strict=variant_structured_strict,
        gene_to_product=gene_to_product,
    )

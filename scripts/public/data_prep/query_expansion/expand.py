from __future__ import annotations

from typing import Dict, List, Tuple

from .table import EntityIndex


def detect_genes(query: str, index: EntityIndex) -> Dict[str, str]:
    """
    Detect entities in query by:
      - entity ID pattern (config entity_id_pattern) when the id exists in the table
      - token match against alias_to_gene (case-insensitive)
    Returns dict: {entity_id_upper -> first_mention_display}
    """
    if query is None:
        return {}

    q = str(query)
    toks = index.token_re.findall(q)
    toks_l = [t.lower() for t in toks]

    detected: Dict[str, str] = {}
    known = set(index.gene_to_product.keys())

    if index.ddb_re is not None:
        for m in index.ddb_re.finditer(q):
            gid_u = m.group(0).upper()
            if gid_u in known and gid_u not in detected:
                detected[gid_u] = gid_u

    for tl in toks_l:
        gid_u = index.alias_to_gene.get(tl)
        if gid_u and gid_u not in detected:
            detected[gid_u] = tl.capitalize()

    return detected


def expand_query_structured(
    query: str,
    index: EntityIndex,
    variant_key: str,
) -> Tuple[str, List[str], str]:
    """
    Tiered structured expansion matching notebooks/query_expansion_bm25.py.

    Returns (expanded_query, detected_entity_ids_sorted, structured_suffix).
    """
    if variant_key not in index.cfg.expand_variants:
        raise KeyError(f"unknown variant: {variant_key}")

    if query is None:
        return ("", [], "")

    cfg = index.cfg
    gene_to_aliases = index.variant_gene_to_aliases[variant_key]
    include_gene_products = index.variant_include_product[variant_key]
    strict = index.variant_structured_strict[variant_key]
    gene_to_product = index.gene_to_product

    q = str(query)
    toks = index.token_re.findall(q)
    present = {t.lower() for t in toks}

    detected_map = detect_genes(q, index)
    detected = sorted(detected_map.keys())
    n_detected = len(detected)

    def _canonical(gid_u: str) -> str:
        aliases_list = gene_to_aliases.get(gid_u, [])
        if aliases_list:
            return aliases_list[0]
        return detected_map[gid_u]

    # 4+ entities: minimal
    if n_detected >= 4:
        if strict:
            return (q, detected, "")
        gid_u = detected[0]
        first_mention = detected_map[gid_u]
        canonical = _canonical(gid_u)
        if canonical.lower().startswith("ddb_g"):
            return (q, detected, "")
        expansion_str = canonical
        if include_gene_products:
            product = gene_to_product.get(gid_u)
            if product:
                expansion_str += ", " + product
        block = f"{first_mention}: {expansion_str}"
        return (q + "\n" + block, detected, block)

    # 3 entities: light (canonical name only)
    if n_detected == 3:
        blocks: List[str] = []
        for gid_u in detected:
            first_mention = detected_map[gid_u]
            canonical = _canonical(gid_u)
            if canonical.lower().startswith("ddb_g"):
                continue
            expansion_str = canonical
            if not strict and include_gene_products:
                product = gene_to_product.get(gid_u)
                if product:
                    expansion_str += ", " + product
            blocks.append(f"{first_mention}: {expansion_str}")
        structured_suffix = (" ||| ".join(blocks)).rstrip() if blocks else ""
        bm25_query = q if not structured_suffix else (q + "\n" + structured_suffix)
        return (bm25_query, detected, structured_suffix)

    # 1–2 entities: full expansion
    blocks = []
    for gid_u in detected:
        first_mention = detected_map[gid_u]
        aliases_list = gene_to_aliases.get(gid_u, [])
        expansion_aliases: List[str] = []
        for a in aliases_list:
            al = a.lower()
            if al not in present and not al.startswith("ddb_g"):
                expansion_aliases.append(a)
        expansion_str = ", ".join(expansion_aliases)
        if include_gene_products:
            product = gene_to_product.get(gid_u)
            product_str = product if product else ""
            if product_str:
                if expansion_str:
                    expansion_str += ", " + product_str
                else:
                    expansion_str = product_str
        if expansion_str:
            blocks.append(f"{first_mention}: {expansion_str}")
    structured_suffix = (" ||| ".join(blocks)).rstrip() if blocks else ""
    bm25_query = q if not structured_suffix else (q + "\n" + structured_suffix)
    return (bm25_query, detected, structured_suffix)

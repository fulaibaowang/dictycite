# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dicty (Python 3.14 venv)
#     language: python
#     name: dicty-py314
# ---

# %% [markdown]
# # Final Public Export (LLM-labeled goldset)
#
# This notebook builds:
# - Labeled goldset JSON (7a/7b): queries, docs, in-text gene labels (`has_detectable_gene`, `n_detected_genes`, `query_expansion_benchmark` from Gene Name+Synonym+DDB id detection; see `query_expansion_dicty_gene_in_text_detection.yaml`).
# - 7d benchmark JSONL/JSON: only `query_expansion_benchmark=yes` rows, plus `query_expansion_Synonym` and `query_expansion_Synonym_Products` from the full `query_expansion_dicty_gene.example.yaml` + `expand_query_structured` (not the same as detection-only alias map).
# - EPMC documents JSONL (7c).
#
# Inputs:
# - output/dicty_gold_build/5a_gold_query_expand.parquet (columns: query, docs)
# - output/dicty_gold_build/4a_claim_groups.parquet (rep claim + gene_id per group_claim_id)
# - dictybase_files/gene_information.txt
# - scripts/public/data_prep/conf/query_expansion_dicty_gene_in_text_detection.yaml (detection)
# - scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml (7d expansion strings)
# - output/dicty_gold_build/6d_llm_full_agreement.tsv
# - output/dicty_gold_build/3_articles_cleaned_abstract.parquet
#
# Outputs: 7a, 7b, 7c as before; and 7d `output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl` (and .json)

# %%
from __future__ import annotations

import sys
from pathlib import Path
import json
from typing import Any, Dict, List

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_PREP = REPO_ROOT / "scripts" / "public" / "data_prep"
if str(_DATA_PREP) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP))
from query_expansion.config import load_expansion_config
from query_expansion.expand import detect_genes, expand_query_structured
from query_expansion.table import build_entity_index


# %% [markdown]
# ## Load inputs

# %%
GOLD_PATH = Path("../output/dicty_gold_build/5a_gold_query_expand.parquet")
CLAIM_GROUPS_PATH = Path("../output/dicty_gold_build/4a_claim_groups.parquet")
GENE_INFO_PATH = (REPO_ROOT / "dictybase_files" / "gene_information.txt").resolve()
LABELS_PATH = Path("../output/dicty_gold_build/6d_llm_full_agreement.tsv")
DOCS_PATH = Path("../output/dicty_gold_build/3_articles_cleaned_abstract.parquet")
OUT_JSONL = Path("../output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl")
OUT_JSON = Path("../output/dicty_gold_build/7a_dicty_gold_llm_public.json")
OUT_JSONL_PRIVATE = Path("../output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl")
OUT_JSON_PRIVATE = Path("../output/dicty_gold_build/7b_dicty_gold_llm_private.json")
DOCS_JSONL_OUT = Path("../output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl")
OUT_7D_JSONL = Path("../output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl")
OUT_7D_JSON = Path("../output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.json")
OUT_7E_JSONL = Path("../output/dicty_gold_build/7e_dicty_gold_query_expansion_benchmark.jsonl")
OUT_7E_JSON = Path("../output/dicty_gold_build/7e_dicty_gold_query_expansion_benchmark.json")
_DET_CONFIG = _DATA_PREP / "conf" / "query_expansion_dicty_gene_in_text_detection.yaml"
_EXPAND_CONFIG = _DATA_PREP / "conf" / "query_expansion_dicty_gene.example.yaml"
DETECTION_INDEX = build_entity_index(GENE_INFO_PATH, load_expansion_config(_DET_CONFIG))
EXPAND_INDEX = build_entity_index(GENE_INFO_PATH, load_expansion_config(_EXPAND_CONFIG))

def load_labels(path: Path) -> pl.DataFrame:
    if path.suffix == ".jsonl":
        df = pl.read_ndjson(path)
    elif path.suffix == ".tsv":
        df = pl.read_csv(path, separator="\t")
    else:
        raise ValueError(f"Unsupported labels format: {path.suffix}")

    if "reason" not in df.columns:
        df = df.with_columns(pl.lit("").alias("reason"))

    return df.with_columns([
        pl.col("group_claim_id").cast(pl.Utf8),
        pl.col("pmid").cast(pl.Utf8),
    ])


def load_gene_lookup(path: Path) -> Dict[str, Dict[str, str]]:
    """gene_id (DDB_G...) -> name, synonyms, products from dictyBase gene_information."""
    df = pl.read_csv(path, separator="\t", infer_schema_length=20_000)
    df = df.rename(
        {
            "GENE ID": "gene_id",
            "Gene Name": "gene_name",
            "Synonyms": "synonyms",
            "Gene products": "gene_products",
        }
    )
    out: Dict[str, Dict[str, str]] = {}
    for row in df.iter_rows(named=True):
        gid = str(row.get("gene_id") or "").strip()
        if not gid:
            continue
        out[gid] = {
            "gene_id": gid,
            "gene_name": str(row.get("gene_name") or "").strip(),
            "synonyms": str(row.get("synonyms") or "").strip(),
            "gene_products": str(row.get("gene_products") or "").strip(),
        }
    return out


def genes_for_gene_id_csv(gene_csv: str, lookup: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    records: List[Dict[str, str]] = []
    for part in str(gene_csv or "").split(","):
        gid = part.strip()
        if not gid or gid in seen:
            continue
        seen.add(gid)
        rec = lookup.get(gid)
        if rec is not None:
            records.append(dict(rec))
        else:
            records.append(
                {
                    "gene_id": gid,
                    "gene_name": "",
                    "synonyms": "",
                    "gene_products": "",
                }
            )
    return records


def build_claim_group_metadata(claim_path: Path) -> Dict[str, Any]:
    """
    group_claim_id str -> {rep_claim_id, gene_id, claim_ids}
    """
    df = pl.read_parquet(claim_path)
    rep = (
        df.filter(pl.col("is_representative_claim"))
        .select(
            [
                pl.col("group_claim_id").cast(pl.Utf8),
                pl.col("rep_claim_id"),
                pl.col("gene_id").cast(pl.Utf8),
            ]
        )
        .unique(subset=["group_claim_id"], keep="first")
    )
    by_group: Dict[str, Any] = {
        r["group_claim_id"]: {
            "rep_claim_id": int(r["rep_claim_id"]),
            "gene_id": str(r.get("gene_id") or "").strip(),
        }
        for r in rep.to_dicts()
    }
    claims = df.group_by(pl.col("group_claim_id").cast(pl.Utf8)).agg(
        pl.col("claim_id").unique().sort().alias("claim_ids")
    )
    for r in claims.to_dicts():
        g = str(r["group_claim_id"])
        ids = r.get("claim_ids") or []
        claim_list = [int(c) for c in ids] if isinstance(ids, list) else []
        if g in by_group:
            by_group[g]["claim_ids"] = claim_list
        else:
            by_group[g] = {
                "rep_claim_id": None,
                "gene_id": "",
                "claim_ids": claim_list,
            }
    for g, v in by_group.items():
        v.setdefault("claim_ids", [])
    return by_group


GENE_LOOKUP = load_gene_lookup(GENE_INFO_PATH)
CLAIM_BY_GROUP = build_claim_group_metadata(CLAIM_GROUPS_PATH)

gold = pl.read_parquet(GOLD_PATH)
labels = load_labels(LABELS_PATH).unique(subset=["group_claim_id", "pmid"])

gold.head(2)


# %% [markdown]
# ## Build final public JSON
#
# We join LLM labels to the goldset and keep one JSON output for public release.

# %%
if "docs" not in gold.columns:
    raise ValueError("Expected 'docs' column in gold_with_query_expand.parquet")

gold_long = (
    gold.select(["group_claim_id", "query", "docs"])
    .explode("docs")
    .with_columns([
        pl.col("docs").struct.field("publication_id").alias("publication_id"),
        pl.col("docs").struct.field("pmid").cast(pl.Utf8).alias("pmid"),
        pl.col("docs").struct.field("title").alias("title"),
        pl.col("docs").struct.field("abstract_clean").alias("abstract_clean"),
        pl.col("docs").struct.field("year").alias("year"),
        pl.col("docs").struct.field("anchor_pos").alias("anchor_pos"),
        pl.col("docs").struct.field("citation_captions").alias("citation_captions"),
    ])
    .drop("docs")
    .with_columns([
        pl.col("group_claim_id").cast(pl.Utf8),
        pl.col("pmid").cast(pl.Utf8),
    ])
 )

labeled = gold_long.join(labels, on=["group_claim_id", "pmid"], how="inner")

grouped = labeled.group_by("group_claim_id").agg([
    pl.first("query").alias("query"),
    pl.struct([
        "publication_id",
        "pmid",
        "title",
        "abstract_clean",
        "year",
        "anchor_pos",
        "citation_captions",
        "doc_match",
        "evidence_level",
        "reason",
    ]).alias("docs"),
])

PUBMED_URL_PREFIX = "http://www.ncbi.nlm.nih.gov/pubmed/"

def _gene_has_expansion_content(gid: str, gene_lookup: Dict[str, Dict[str, str]]) -> bool:
    rec = gene_lookup.get(gid, {})
    syns = str(rec.get("synonyms") or "").strip()
    prod = str(rec.get("gene_products") or "").strip()
    return bool(syns and syns.upper() != "NA" and prod and prod.upper() != "NA")


questions = grouped.sort("group_claim_id").to_dicts()
for q in questions:
    pmids = [d.get("pmid") for d in q.get("docs", []) if d.get("pmid")]
    q["pmids"] = pmids
    # Canonical pipeline keys
    gcid = str(q.get("group_claim_id", ""))
    q["query_id"] = gcid
    q["query_text"] = (q.get("query") or "").strip()
    q["documents"] = [PUBMED_URL_PREFIX + str(p) for p in pmids if p]
    meta = CLAIM_BY_GROUP.get(gcid, {})
    rep_cid = meta.get("rep_claim_id")
    q["rep_claim_id"] = int(rep_cid) if rep_cid is not None else None
    gcsv = str(meta.get("gene_id") or "")
    q["gene_id"] = gcsv
    q["genes"] = genes_for_gene_id_csv(gcsv, GENE_LOOKUP)
    q["claim_ids"] = meta.get("claim_ids", [])
    _qt = q["query_text"]
    _detected = detect_genes(_qt, DETECTION_INDEX)
    n_det = len(_detected)
    _expandable = (
        n_det == 1 and all(_gene_has_expansion_content(gid, GENE_LOOKUP) for gid in _detected)
    )
    _detected_ids = sorted(_detected.keys())
    if _expandable:
        _syn, _, _ = expand_query_structured(_qt, EXPAND_INDEX, "synonyms_only")
        _long, _, _ = expand_query_structured(_qt, EXPAND_INDEX, "long")
        _benchmark = bool(_syn[len(_qt):].strip()) and bool(_long[len(_qt):].strip())
    else:
        _benchmark = False
    q["query_gene_expansion"] = {
        "has_detectable_gene": "yes" if n_det > 0 else "no",
        "n_detected_genes": n_det,
        "detected_gene_ids": _detected_ids,
        "detected_gene_expandable": "yes" if _expandable else "no",
        "query_expansion_benchmark": "yes" if _benchmark else "no",
    }
    # Replace genes with detected gene records whenever detection succeeds.
    # The query text is the ground truth for what gene the query is about.
    # Only fall back to the DictyBase annotation when nothing is detectable (n_det == 0).
    if n_det > 0:
        q["genes"] = [
            GENE_LOOKUP.get(gid, {"gene_id": gid, "gene_name": "", "synonyms": "", "gene_products": ""})
            for gid in _detected_ids
        ]

def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Full payload (all fields) → private
_write_jsonl(OUT_JSONL_PRIVATE, questions)
print(f"Saved (private): {OUT_JSONL_PRIVATE}")

# Public: query_id, query_text, genes, documents, docs, in-text gene labels
def to_public_question(q: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": q["query_id"],
        "query_text": q["query_text"],
        "genes": q.get("genes", []),
        "documents": q["documents"],
        "docs": q.get("docs", []),
        "query_gene_expansion": q.get("query_gene_expansion", {
            "has_detectable_gene": "no",
            "n_detected_genes": 0,
            "detected_gene_expandable": "no",
            "query_expansion_benchmark": "no",
        }),
    }

questions_public = [to_public_question(q) for q in questions]
_write_jsonl(OUT_JSONL, questions_public)
print(f"Saved (public): {OUT_JSONL}")
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"questions": questions_public}, f, ensure_ascii=False, indent=2)
print(f"Saved (public JSON): {OUT_JSON}")
with open(OUT_JSON_PRIVATE, "w", encoding="utf-8") as f:
    json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)
print(f"Saved (private JSON): {OUT_JSON_PRIVATE}")

# 7d: only query_expansion_benchmark=yes; add full structured expansion (example.yaml)
def _add_expansion_fields(
    q_pub: Dict[str, Any],
    expand_index,
    gene_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    qt = str(q_pub.get("query_text") or "")
    syn, detected_ids, _ = expand_query_structured(qt, expand_index, "synonyms_only")
    long_q, _, _ = expand_query_structured(qt, expand_index, "long")
    out = dict(q_pub)
    out.pop("genes", None)
    qge = dict(out.get("query_gene_expansion") or {})
    qge["detected_gene_ids"] = detected_ids
    qge["detected_genes"] = [
        gene_lookup.get(gid, {"gene_id": gid, "gene_name": "", "synonyms": "", "gene_products": ""})
        for gid in detected_ids
    ]
    qge["expansion_synonym"] = syn
    qge["expansion_synonym_products"] = long_q
    out["query_gene_expansion"] = qge
    return out


benchmark_7d_candidates = [
    _add_expansion_fields(q, EXPAND_INDEX, GENE_LOOKUP)
    for q in questions_public
    if q.get("query_gene_expansion", {}).get("query_expansion_benchmark") == "yes"
]
def _both_expansions_nonempty(r: Dict[str, Any]) -> bool:
    qt = r.get("query_text", "")
    qge = r.get("query_gene_expansion", {})
    syn_suffix = qge.get("expansion_synonym", "")[len(qt):]
    long_suffix = qge.get("expansion_synonym_products", "")[len(qt):]
    return bool(syn_suffix.strip()) and bool(long_suffix.strip())


benchmark_7d = [
    r for r in benchmark_7d_candidates
    if r.get("query_gene_expansion", {}).get("detected_gene_ids")
    and all(
        _gene_has_expansion_content(gid, GENE_LOOKUP)
        for gid in r["query_gene_expansion"]["detected_gene_ids"]
    )
    and _both_expansions_nonempty(r)
]
n_dropped = len(benchmark_7d_candidates) - len(benchmark_7d)
print(f"7d filter: kept {len(benchmark_7d)}, dropped {n_dropped} (no synonyms/product in DB or synonyms already in query)")
_write_jsonl(OUT_7D_JSONL, benchmark_7d)
print(f"Saved (7d benchmark): {OUT_7D_JSONL}  (n={len(benchmark_7d)})")
with open(OUT_7D_JSON, "w", encoding="utf-8") as f:
    json.dump({"questions": benchmark_7d}, f, ensure_ascii=False, indent=2)
print(f"Saved (7d JSON): {OUT_7D_JSON}")

# 7e: same as 7d but drop queries that include any doc with evidence_level=needs_fulltext (abstract-eval subset)
def _no_needs_fulltext_doc(q: Dict[str, Any]) -> bool:
    for d in q.get("docs") or []:
        if (d.get("evidence_level") or "").strip() == "needs_fulltext":
            return False
    return True


benchmark_7e = [q for q in benchmark_7d if _no_needs_fulltext_doc(q)]
n_7e_dropped = len(benchmark_7d) - len(benchmark_7e)
print(f"7e filter: kept {len(benchmark_7e)}, dropped {n_7e_dropped} (any doc needs_fulltext)")
_write_jsonl(OUT_7E_JSONL, benchmark_7e)
print(f"Saved (7e benchmark): {OUT_7E_JSONL}  (n={len(benchmark_7e)})")
with open(OUT_7E_JSON, "w", encoding="utf-8") as f:
    json.dump({"questions": benchmark_7e}, f, ensure_ascii=False, indent=2)
print(f"Saved (7e JSON): {OUT_7E_JSON}")


# %% [markdown]
# ## Quick sanity checks

# %%
total_questions = len(questions)
total_docs = sum(len(q.get("docs", [])) for q in questions)

print(f"Questions: {total_questions}")
print(f"Labeled docs: {total_docs}")
print(f"Label source: {LABELS_PATH}")


# %% [markdown]
# ## Label stats
#
# Percent breakdown for `doc_match` and `evidence_level` over labeled pairs.

# %%
def show_label_stats(df: pl.DataFrame, col: str) -> None:
    total = df.height
    if total == 0:
        print(f"{col}: no rows")
        return
    counts = (
        df.group_by(col)
        .len()
        .sort("len", descending=True)
        .with_columns((pl.col("len") / total * 100).round(2).alias("pct"))
    )
    print(f"\n{col} (n={total})")
    print(counts)

show_label_stats(labeled, "doc_match")
show_label_stats(labeled, "evidence_level")

# %% [markdown]
# ## Export EPMC documents JSONL
#
# Writes one JSON object per line from the cleaned EPMC abstracts parquet (key `abstract` = cleaned text).

# %%
# Export with key "abstract" (value = abstract_clean) so consumers (e.g. index scripts) use one field name.
docs = (
    pl.read_parquet(DOCS_PATH)
    .with_columns(pl.col("pmid").cast(pl.Utf8))
    .rename({"abstract_clean": "abstract"})
)
docs.write_ndjson(DOCS_JSONL_OUT)
print(f"Saved: {DOCS_JSONL_OUT}")

# %%

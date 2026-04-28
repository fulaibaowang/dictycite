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
# This notebook builds two public JSON artifacts:
# - Labeled goldset (queries + labeled docs).
# - EPMC documents JSONL (title/abstract store).
#
# Inputs:
# - output/dicty_gold_build/5a_gold_query_expand.parquet (columns: query, docs)
# - output/dicty_gold_build/4a_claim_groups.parquet (rep claim + gene_id per group_claim_id)
# - dictybase_files/gene_information.txt (gene name, synonyms, products for each DDB_G id)
# - output/dicty_gold_build/6d_llm_full_agreement.tsv
# - output/dicty_gold_build/3_articles_cleaned_abstract.parquet
#
# Output JSONL per question: 7a has query_id, query_text, genes, documents, docs (gene DDB_G ids live under genes[].gene_id).
# 7b private also: rep_claim_id, gene_id (CSV), group_claim_id, query, claim_ids, pmids.
#
# Outputs:
# - output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl / .json (full payload, all fields)
# - output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl / .json (canonical pipeline keys)
# - output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl

# %%
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

import polars as pl


# %% [markdown]
# ## Load inputs

# %%
GOLD_PATH = Path("../output/dicty_gold_build/5a_gold_query_expand.parquet")
CLAIM_GROUPS_PATH = Path("../output/dicty_gold_build/4a_claim_groups.parquet")
GENE_INFO_PATH = Path("../dictybase_files/gene_information.txt")
LABELS_PATH = Path("../output/dicty_gold_build/6d_llm_full_agreement.tsv")
DOCS_PATH = Path("../output/dicty_gold_build/3_articles_cleaned_abstract.parquet")
OUT_JSONL = Path("../output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl")
OUT_JSON = Path("../output/dicty_gold_build/7a_dicty_gold_llm_public.json")
OUT_JSONL_PRIVATE = Path("../output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl")
OUT_JSON_PRIVATE = Path("../output/dicty_gold_build/7b_dicty_gold_llm_private.json")
DOCS_JSONL_OUT = Path("../output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl")

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

def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Full payload (all fields) → private
_write_jsonl(OUT_JSONL_PRIVATE, questions)
print(f"Saved (private): {OUT_JSONL_PRIVATE}")

# Public: query_id, query_text, genes, documents, docs
def to_public_question(q: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": q["query_id"],
        "query_text": q["query_text"],
        "genes": q.get("genes", []),
        "documents": q["documents"],
        "docs": q.get("docs", []),
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

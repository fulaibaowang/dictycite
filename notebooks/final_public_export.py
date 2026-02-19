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
# - output/cleaned/gold_with_query_expand.parquet (columns: query, query_expand_synonyms, query_expand_long, docs; or query, query_expand, docs for backward compat)
# - output/llama_full_agreement_cases.tsv (or output/llm_labels_*.jsonl)
# - output/cleaned/articles_all_cleaned_abstract.parquet
#
# Output JSON fields per question: id, original_query, body_expansion_synonyms, body_expansion_long, body (= original_query), documents, docs.
#
# Outputs:
# - output/cleaned/dicty_gold_llm_private.json (full payload, all fields)
# - output/cleaned/dicty_gold_llm_public.json (clean, BioASQ-style keys only)
# - output/cleaned/articles_all_cleaned_abstract.jsonl

# %%
from pathlib import Path
import json
import polars as pl


# %% [markdown]
# ## Load inputs

# %%
GOLD_PATH = Path("../output/cleaned/gold_with_query_expand.parquet")
LABELS_PATH = Path("../output/llama_full_agreement_cases.tsv")
DOCS_PATH = Path("../output/cleaned/articles_all_cleaned_abstract.parquet")
OUT_JSON = Path("../output/cleaned/dicty_gold_llm_public.json")
OUT_JSON_PRIVATE = Path("../output/cleaned/dicty_gold_llm_private.json")
DOCS_JSONL_OUT = Path("../output/cleaned/articles_all_cleaned_abstract.jsonl")

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

# Prefer query_expand_synonyms / query_expand_long; fallback to query_expand for backward compat
select_cols = ["group_claim_id", "query", "docs"]
if "query_expand_synonyms" in gold.columns and "query_expand_long" in gold.columns:
    select_cols.extend(["query_expand_synonyms", "query_expand_long"])
else:
    gold = gold.with_columns([
        pl.col("query_expand").alias("query_expand_synonyms"),
        pl.col("query_expand").alias("query_expand_long"),
    ])
    select_cols.extend(["query_expand_synonyms", "query_expand_long"])

gold_long = (
    gold.select(select_cols)
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
    pl.first("query_expand_synonyms").alias("query_expand_synonyms"),
    pl.first("query_expand_long").alias("query_expand_long"),
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
    # BioASQ-like fields: body = original_query; expansion variants for retrieval --query-field
    q["id"] = str(q.get("group_claim_id", ""))
    q["original_query"] = (q.get("query") or "").strip()
    q["body"] = q["original_query"]
    q["body_expansion_synonyms"] = (q.get("query_expand_synonyms") or "").strip()
    q["body_expansion_long"] = (q.get("query_expand_long") or "").strip()
    q["documents"] = [PUBMED_URL_PREFIX + str(p) for p in pmids if p]

# Full payload (all fields) → private
OUT_JSON_PRIVATE.write_text(json.dumps({"questions": questions}, indent=2), encoding="utf-8")
print(f"Saved (private): {OUT_JSON_PRIVATE}")

# Clean public: BioASQ-style key fields + expansion variants
def to_public_question(q):
    return {
        "id": q["id"],
        "original_query": q["original_query"],
        "body_expansion_synonyms": q["body_expansion_synonyms"],
        "body_expansion_long": q["body_expansion_long"],
        "body": q["body"],
        "documents": q["documents"],
        "docs": q.get("docs", []),
    }

questions_public = [to_public_question(q) for q in questions]
OUT_JSON.write_text(json.dumps({"questions": questions_public}, indent=2), encoding="utf-8")
print(f"Saved (public): {OUT_JSON}")


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

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
# - output/dicty_gold_build/5a_gold_query_expand.parquet (columns: query, query_expand_synonyms, query_expand_long, docs; or query, query_expand, docs for backward compat)
# - output/dicty_gold_build/6d_llm_full_agreement.tsv
# - output/dicty_gold_build/3_articles_cleaned_abstract.parquet
#
# Output JSONL fields per question: query_id, query_text, query_text_expansion_synonyms, query_text_expansion_long, documents, docs.
#
# Outputs:
# - output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl (full payload, all fields)
# - output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl (canonical pipeline keys)
# - output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl

# %%
from pathlib import Path
import json
import polars as pl


# %% [markdown]
# ## Load inputs

# %%
GOLD_PATH = Path("../output/dicty_gold_build/5a_gold_query_expand.parquet")
LABELS_PATH = Path("../output/dicty_gold_build/6d_llm_full_agreement.tsv")
DOCS_PATH = Path("../output/dicty_gold_build/3_articles_cleaned_abstract.parquet")
OUT_JSONL = Path("../output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl")
OUT_JSONL_PRIVATE = Path("../output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl")
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
    # Canonical pipeline keys
    q["query_id"] = str(q.get("group_claim_id", ""))
    q["query_text"] = (q.get("query") or "").strip()
    q["query_text_expansion_synonyms"] = (q.get("query_expand_synonyms") or "").strip()
    q["query_text_expansion_long"] = (q.get("query_expand_long") or "").strip()
    q["documents"] = [PUBMED_URL_PREFIX + str(p) for p in pmids if p]

def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Full payload (all fields) → private
_write_jsonl(OUT_JSONL_PRIVATE, questions)
print(f"Saved (private): {OUT_JSONL_PRIVATE}")

# Clean public: query_id, query_text, query_text_expansion_*, documents, docs
def to_public_question(q):
    return {
        "query_id": q["query_id"],
        "query_text": q["query_text"],
        "query_text_expansion_synonyms": q["query_text_expansion_synonyms"],
        "query_text_expansion_long": q["query_text_expansion_long"],
        "documents": q["documents"],
        "docs": q.get("docs", []),
    }

questions_public = [to_public_question(q) for q in questions]
_write_jsonl(OUT_JSONL, questions_public)
print(f"Saved (public): {OUT_JSONL}")


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

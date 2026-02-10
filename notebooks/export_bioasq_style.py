# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Export DictyCite as BioASQ-style JSON
#
# This notebook builds:
# - A BioASQ-style questions JSON (queries + gold PMIDs + metadata)
# - A JSONL document store (title/abstract from EPMC)
#
# Inputs:
# - output/cleaned/gold_with_query_expand.parquet
# - output/cleaned/articles_all_cleaned_abstract.parquet

# %%
from pathlib import Path
import json
import polars as pl


# %% [markdown]
# ## Load inputs

# %%
GOLD_PATH = Path("../output/cleaned/gold_with_query_expand.parquet")
DOCS_PATH = Path("../output/cleaned/articles_all_cleaned_abstract.parquet")

gold = pl.read_parquet(GOLD_PATH)
docs = pl.read_parquet(DOCS_PATH)

gold.head(2)


# %% [markdown]
# ## Build BioASQ-style questions JSON
#
# We keep all available metadata from `gold_with_query_expand` and attach the gold documents.

# %%
# Normalize docs list (if present) to a flat dataframe for join and regroup
if "docs" in gold.columns:
    gold_long = (
        gold.select(["group_claim_id", "query", "query_expand", "docs"])
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
    )
else:
    gold_long = gold

# Build BioASQ-style questions
questions = []
for row in gold_long.group_by("group_claim_id").agg([
    pl.first("query").alias("query"),
    pl.first("query_expand").alias("query_expand"),
    pl.col("pmid").drop_nulls().unique().sort().alias("pmids"),
    pl.col("publication_id").drop_nulls().unique().sort().alias("publication_ids"),
    pl.col("title").drop_nulls().unique().alias("titles"),
    pl.col("abstract_clean").drop_nulls().unique().alias("abstracts"),
    pl.col("year").drop_nulls().unique().sort().alias("years"),
    pl.col("anchor_pos").drop_nulls().alias("anchor_pos"),
    pl.col("citation_captions").drop_nulls().alias("citation_captions"),
]).iter_rows(named=True):
    qid = str(row["group_claim_id"])
    pmid_list = [str(p) for p in row.get("pmids", [])]
    doc_urls = [f"http://www.ncbi.nlm.nih.gov/pubmed/{p}" for p in pmid_list]

    questions.append({
        "id": qid,
        "body": row.get("query", ""),
        "body_expand": row.get("query_expand", ""),
        "documents": doc_urls,
        "pmids": pmid_list,
        "publication_ids": row.get("publication_ids", []),
        "titles": row.get("titles", []),
        "abstracts": row.get("abstracts", []),
        "years": row.get("years", []),
        "anchor_pos": row.get("anchor_pos", []),
        "citation_captions": row.get("citation_captions", []),
    })

bioasq_json = {"questions": questions}
OUT_JSON = Path("../output/cleaned/dicty_bioasq_style.json")
OUT_JSON.write_text(json.dumps(bioasq_json, indent=2), encoding="utf-8")
print(f"Saved: {OUT_JSON}")


# %% [markdown]
# ## Build document JSONL from EPMC table
#
# This mirrors BioASQ's `subset_pubmed.jsonl` style, with one JSON per PMID.

# %%
DOCS_OUT = Path("../output/cleaned/articles_all_cleaned_abstract.jsonl")

# Keep all available fields
cols = docs.columns
docs_jsonl = docs.select(cols).with_columns(
    pl.col("pmid").cast(pl.Utf8)
)

with DOCS_OUT.open("w", encoding="utf-8") as f:
    for row in docs_jsonl.iter_rows(named=True):
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

print(f"Saved: {DOCS_OUT}")


# Data Prep Workflow (Output Paths)

This file lists intermediate output paths referenced in the data prep workflow.

## 1) Curator Notes → Claims + dictybase publication_id

Outputs:
- `output/curator_notes.parquet`
- `output/curator_claims.parquet`
- `output/publications.parquet`

## 2) Publication Mapping → PMID

Output:
- `output/gene_publication_pmid.parquet`

## 3) EPMC/PubMed Fetch (title + abstract)

Output:
- `output/cleaned/articles_all_cleaned_abstract.parquet` — source of truth (column `abstract_clean`).
- `output/cleaned/articles_all_cleaned_abstract.json` — written by final_public_export (key **`abstract`** = cleaned text).
- `output/cleaned/articles_all_cleaned_abstract.jsonl` — not written by the repo. Generate from the parquet (e.g. Polars `write_ndjson`) with `abstract_clean` aliased to **`abstract`** so index scripts (`build_bm25_index_from_jsonl_shards.py`, etc.) that expect key `abstract` work.

## 4) Merge + Clean (claims ↔ PMIDs ↔ abstracts)

Key outputs:
- `output/cleaned/claim_cleaned_long_pmids_nonNA.parquet`
- `output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet`

Query expansion output:
- `output/cleaned/gold_with_query_expand.parquet`
- `output/cleaned/gold_with_query_expand_flat.tsv`

Gold review outputs:
- `output/cleaned/golden_grouped.parquet`
- `tmp/golden_flat.tsv`

## 5) Goldset Labeling (LLM review)

Output:
- `output/llm_labels_*.jsonl`

## Public release

- `output/cleaned/dicty_gold_llm_public.json`
- `output/cleaned/articles_all_cleaned_abstract.json`

# Data Prep Workflow (Output Paths)

This file lists output paths referenced in the data prep workflow.

All versioned pipeline artifacts live under **`output/dicty_gold_build/`** (numbered `N_…` filenames). Batch execution of `datasets_merge_clean` via nbconvert/jupytext may require setting **`DICTYCITE_ROOT`** to the repository root if the Jupyter kernel’s working directory is not the repo.

## 1) Curator Notes → Claims + dictybase publication_id

Output:
- `output/dicty_gold_build/1_curator_claims.parquet`

## 2) Publication Mapping → PMID

Output:
- `output/dicty_gold_build/2_publication_id_pmid.csv`

## 3) EPMC/PubMed Fetch (title + abstract)

Output:
- `output/dicty_gold_build/3_articles_cleaned_abstract.parquet`

## 4) Merge + Clean (claims ↔ PMIDs ↔ abstracts)

Key outputs:
- `output/dicty_gold_build/4a_claim_groups.parquet` (claim ↔ group + variant text + gene_id + canonical query)
- `output/dicty_gold_build/4b_golden_grouped.parquet`

## 5) Query Expansion

Output:
- `output/dicty_gold_build/5a_gold_query_expand.parquet`
- `output/dicty_gold_build/5b_gold_query_expand_flat.tsv`

## 6) Goldset Labeling (LLM review)

Output:
- `output/dicty_gold_build/6a_llm_labels_run1.jsonl`
- `output/dicty_gold_build/6b_llm_labels_run2.jsonl`
- `output/dicty_gold_build/6c_llm_labels_run3.jsonl`
- `output/dicty_gold_build/6d_llm_full_agreement.tsv`

## Public release

- `output/dicty_gold_build/7a_dicty_gold_llm_public.json`
- `output/dicty_gold_build/7b_dicty_gold_llm_private.json`
- `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl`

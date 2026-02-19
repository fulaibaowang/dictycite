# Data Prep Workflow

This document describes how the dataset is assembled and cleaned. The goal is a clean mapping:

**Claims (Dictybase curator notes)**
→ **dictybase publication_id**
→ **PMID**
→ **title + abstract**

## 1) Curator Notes → Claims + dictybase publication_id

Download curator notes and extract claim sentences with citation anchors.

- Production script: [scripts/public/data_prep/dicty_curator_notes.py](../scripts/public/data_prep/dicty_curator_notes.py)
- Notebook: [notebooks/curator_notes_download.ipynb](../notebooks/curator_notes_download.ipynb)

## 2) Publication Mapping → PMID

Build the mapping between dictybase publication_id and PMID by looping over all genes and parsing the references tab.

- Production script: [scripts/public/data_prep/dicty_publication.py](../scripts/public/data_prep/dicty_publication.py)
- Notebook: [notebooks/gene_publication_mapping.ipynb](../notebooks/gene_publication_mapping.ipynb)

## 3) EPMC/PubMed Fetch (title + abstract)

Fetch dicty literature from EPMC and normalize the abstracts.

- Workflow guide: [scripts/public/article_fetching/README.md](../scripts/public/article_fetching/README.md)
- Exploration notebook: [notebooks/epmc_fetch_exploration.ipynb](../notebooks/epmc_fetch_exploration.ipynb)

## 4) Merge + Clean (claims ↔ PMIDs ↔ abstracts)

This step merges curator claims with publication IDs, then joins the PMID mapping and EPMC abstracts.
It also performs cleaning and de-duplication to produce a clean dataset for downstream use.

- Notebook: [notebooks/datasets_merge_clean.ipynb](../notebooks/datasets_merge_clean.ipynb)

Cleaning summary (high level):
- Remove very short claims.
- Merge exact duplicates and inconsistent citation variants.
- Normalize punctuation and whitespace.
- Cluster near-duplicate claims using TF-IDF similarity and group them.
- Produce a grouped gold dataset for review.

## 4.5) Query Expansion (for goldset)

We build a query-expanded goldset that appends structured gene aliases/products to the original claim.
Two variants are produced: **query_expand_synonyms** (gene name + synonyms only) and **query_expand_long** (synonyms + gene products). Gene IDs (DDB_G…) in the query trigger expansion but are never appended to the expansion text.

- Notebook: [notebooks/query_expansion_and_test.ipynb](../notebooks/query_expansion_and_test.ipynb)
- Gene lookup: [dictybase_files/gene_information.txt](../dictybase_files/gene_information.txt) — tab-separated columns **GENE ID**, **Gene Name**, **Synonyms** (comma-separated), **Gene products**. Used only for expansion; ambiguous aliases are skipped.

## 5) Goldset Labeling (LLM review)

Titles and abstracts are not always enough to validate a claim. Some claims need full text or manual review.
To reduce false positives, we run a labeling step to build a higher-quality goldset.

- Production script: [scripts/public/data_prep/dicty_claim_labeler.py](../scripts/public/data_prep/dicty_claim_labeler.py)
- Notebook: [notebooks/goldset_llm_labeling.ipynb](../notebooks/goldset_llm_labeling.ipynb)

Labeling details (summary):
- Input: `output/cleaned/gold_with_query_expand_flat.tsv` (claim + PMID + title/abstract). The labeler uses **query_expand** (long variant: synonyms + gene products) as the claim text when present, otherwise **query**.
- Run the labeler 3 times to measure consistency:
	- `output/llm_labels_goldset_run1.jsonl`
	- `output/llm_labels_goldset_run2.jsonl`
	- `output/llm_labels_goldset_run3.jsonl`
- Compare runs and compute agreement 

## 6) Final Public Export (JSON)

We join the goldset with LLM labels and publish the labeled goldset JSON.
Expects parquet with `query`, `query_expand_synonyms`, `query_expand_long`, and `docs` (or legacy `query_expand` for backward compat).

- Notebook: [notebooks/final_public_export.ipynb](../notebooks/final_public_export.ipynb)
- Outputs:
	- `output/cleaned/dicty_gold_llm_public.json`
	- `output/cleaned/articles_all_cleaned_abstract.jsonl`

Schema (high level):

- `questions[].id`, `body`, `body_expansion_synonyms`, `body_expansion_long`, `documents`, `docs`
- `questions[].docs[]` includes `pmid`, `title`, `abstract_clean`, `year`, `anchor_pos`, `citation_captions`, `doc_match`, `evidence_level`, `reason`

Retrieval scripts (BM25, dense, rerank) accept `--query-field` to choose which text to use (e.g. `body_expansion_long` for BM25, `body` or `body_expansion_synonyms` for rerank).

EPMC JSONL:
- One JSON object per line from `articles_all_cleaned_abstract.parquet` (pmid/title/abstract metadata). Exported with key **`abstract`** (value is the cleaned abstract text).

## Outputs

- `output/cleaned/dicty_gold_llm_public.json`
- `output/cleaned/articles_all_cleaned_abstract.jsonl`

## Summary (short)

We start with Dictybase curator notes and extract claims that cite a dictybase `publication_id`.
Those IDs are mapped to PubMed PMIDs, then joined with EPMC/PubMed title and abstract text.
After cleaning and de-duplication, we label claim-PMID pairs to build a higher-quality goldset
and export the final public JSONs for claims (with labels) and abstracts.



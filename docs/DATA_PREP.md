# Data Prep Workflow

This document describes how the dataset is assembled and cleaned. The goal is a clean mapping:

**Claims (Dictybase curator notes)**
→ **dictybase publication_id**
→ **PMID**
→ **title + abstract**

## 1) Curator Notes → Claims + dictybase publication_id

Download curator notes and extract claim sentences with citation anchors.

- Production script: [scripts/public/dicty_curator_notes.py](../scripts/public/dicty_curator_notes.py)
- Notebook: [notebooks/curator_notes_download.ipynb](../notebooks/curator_notes_download.ipynb)

Outputs:
- `output/curator_notes.parquet`
- `output/curator_claims.parquet`
- `output/publications.parquet`

## 2) Publication Mapping → PMID

Build the mapping between dictybase publication_id and PMID by looping over all genes and parsing the references tab.

- Production script: [scripts/public/dicty_publication.py](../scripts/public/dicty_publication.py)
- Notebook: [notebooks/gene_publication_mapping.ipynb](../notebooks/gene_publication_mapping.ipynb)

Output:
- `output/gene_publication_pmid.parquet`

## 3) EPMC/PubMed Fetch (title + abstract)

Fetch dicty literature from EPMC and normalize the abstracts.

- Workflow guide: [scripts/public/article_fetching/README.md](../scripts/public/article_fetching/README.md)
- Exploration notebook: [notebooks/epmc_fetch_exploration.ipynb](../notebooks/epmc_fetch_exploration.ipynb)

Output:
- `output/cleaned/articles_all_cleaned_abstract.parquet`

## 4) Merge + Clean (claims ↔ PMIDs ↔ abstracts)

This step merges curator claims with publication IDs, then joins the PMID mapping and EPMC abstracts.
It also performs cleaning and de-duplication to produce a clean dataset for downstream use.

- Notebook: [notebooks/datasets_merge_clean.ipynb](../notebooks/datasets_merge_clean.ipynb)

Key outputs:
- `output/cleaned/claim_cleaned_long_pmids_nonNA.parquet`
- `output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet`

### Cleaning summary (high level)

- Remove very short claims.
- Merge exact duplicates and inconsistent citation variants.
- Normalize punctuation and whitespace.
- Cluster near-duplicate claims using TF-IDF similarity and group them.
- Produce a grouped gold dataset:
  - `output/cleaned/golden_grouped.parquet`
  - `tmp/golden_flat.tsv`

## 5) Goldset Labeling (LLM review)

Titles and abstracts are not always enough to validate a claim. Some claims need full text or manual review.
To reduce false positives, we run a labeling step to build a higher-quality goldset.

- Production script: [scripts/public/dicty_claim_labeler.py](../scripts/public/dicty_claim_labeler.py)
- Notebook: [notebooks/goldset_llm_labeling.ipynb](../notebooks/goldset_llm_labeling.ipynb)

Output:
- `output/llm_labels_*.jsonl`

## Next Steps

Retrieval and reranking are planned, but not started. The current focus is a clean, traceable dataset.

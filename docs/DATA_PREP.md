# Data Prep Workflow

This document describes how the dataset is assembled and cleaned. The goal is a clean mapping:

**Claims (Dictybase curator notes)**
→ **dictybase publication_id**
→ **PMID**
→ **title + abstract**

## 1) Curator Notes → Claims + dictybase publication_id

Download curator notes and extract claim sentences with citation anchors.

- Production script: [scripts/public/data_prep/dicty_curator_notes.py](../scripts/public/data_prep/dicty_curator_notes.py)
- Notebook: [notebooks/01_curator_notes_download.ipynb](../notebooks/01_curator_notes_download.ipynb)

## 2) Publication Mapping → PMID

Build the mapping between dictybase publication_id and PMID by looping over all genes and parsing the references tab.

- Production script: [scripts/public/data_prep/dicty_publication.py](../scripts/public/data_prep/dicty_publication.py)
- Notebook: [notebooks/02_gene_publication_mapping.ipynb](../notebooks/02_gene_publication_mapping.ipynb)

## 3) EPMC/PubMed Fetch (title + abstract)

Fetch dicty literature from EPMC and normalize the abstracts.

- Workflow guide: [scripts/public/article_fetching/README.md](../scripts/public/article_fetching/README.md)
- Exploration notebook: [notebooks/03_epmc_fetch_exploration.ipynb](../notebooks/03_epmc_fetch_exploration.ipynb)

## 4) Merge + Clean (claims ↔ PMIDs ↔ abstracts)

This step merges curator claims with publication IDs, then joins the PMID mapping and EPMC abstracts.
It also performs cleaning and de-duplication to produce a clean dataset for downstream use.

- Notebook: [notebooks/04_datasets_merge_clean.ipynb](../notebooks/04_datasets_merge_clean.ipynb)

Cleaning summary (high level):
- Remove very short claims.
- Merge exact duplicates and inconsistent citation variants.
- Normalize punctuation and whitespace.
- Cluster near-duplicate claims using TF-IDF similarity and group them.
- Produce a grouped gold dataset for review.

Output `4a_claim_groups.parquet` includes two gene-detection columns added by running in-text detection on each group's `canonical_query`:
- `detected_gene_ids` — comma-joined gene IDs found in the query text (empty if none detected).
- `corrected_gene_id` — the single detected gene ID when exactly one gene is found; empty when 0 or 2+ are detected. Provides a text-based correction to the DictyBase-annotated `gene_id` when they differ.

## 4.5) Query Expansion (for goldset)

We build a query-expanded goldset that appends structured gene aliases/products to the original claim.
Two variants are produced: **query_expand_synonyms** (gene name + synonyms only) and **query_expand_long** (synonyms + gene products). Gene IDs (DDB_G…) in the query trigger expansion but are never appended to the expansion text.

Expansion is tiered by how many genes are detected in the query:
- **1–2 genes:** full expansion (canonical names + filtered synonyms; long variant also adds gene products).
- **3 genes:** light expansion — canonical names only (no synonyms), to avoid clutter.
- **4+ genes:** no synonym expansion; query is left as-is or at most one minimal expansion.

The two variants differ in strictness: **query_expand_synonyms** applies these tiers strictly (fewer added terms). **query_expand_long** is looser (e.g. still adds gene products when expansion is light or minimal), giving variety without excessive noise.

- Notebook: [notebooks/05_query_expansion_bm25.ipynb](../notebooks/05_query_expansion_bm25.ipynb) (jupytext: `05_query_expansion_bm25.py`)
- Gene lookup: [dictybase_files/gene_information.txt](../dictybase_files/gene_information.txt) — tab-separated columns **GENE ID**, **Gene Name**, **Synonyms** (comma-separated), **Gene products**. Used only for expansion; ambiguous aliases are skipped.
- Reproducible JSONL enrichment (YAML + table, same tier rules): [scripts/public/data_prep/apply_query_expansion.py](../scripts/public/data_prep/apply_query_expansion.py) with example config [scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml](../scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml). Requires `pip install -r scripts/public/data_prep/requirements-query-expansion.txt`.

## 5) Goldset Labeling (LLM review)

Titles and abstracts are not always enough to validate a claim. Some claims need full text or manual review.
To reduce false positives, we run a labeling step to build a higher-quality goldset.

- Production script: [scripts/public/data_prep/dicty_claim_labeler.py](../scripts/public/data_prep/dicty_claim_labeler.py)
- Notebook: [notebooks/06_goldset_llm_labeling.ipynb](../notebooks/06_goldset_llm_labeling.ipynb)

Labeling details (summary):
- Input: `output/dicty_gold_build/5b_gold_query_expand_flat.tsv` (claim + PMID + title/abstract). The labeler uses **query_expand** (long variant: synonyms + gene products) as the claim text when present, otherwise **query**.
- Run the labeler 3 times to measure consistency:
	- `output/dicty_gold_build/6a_llm_labels_run1.jsonl`
	- `output/dicty_gold_build/6b_llm_labels_run2.jsonl`
	- `output/dicty_gold_build/6c_llm_labels_run3.jsonl`
- Compare runs and compute agreement 

## 6) Final Public Export (JSON)

We join the goldset with LLM labels and publish the labeled goldset JSON.

- Notebook: [notebooks/07_final_public_export.ipynb](../notebooks/07_final_public_export.ipynb)
- Outputs:
	- `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl` / `.json` — full public goldset (1,656 queries)
	- `output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl` / `.json` — full payload including internal fields
	- `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl` — EPMC article records
	- `output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl` / `.json` — query expansion benchmark subset (563 queries)

### 7a schema

Each record has:

```
query_id          string   — group_claim_id
query_text        string   — canonical claim text (the query)
genes             list     — detected gene records when n_detected_genes > 0;
                             DictyBase-annotated gene records otherwise.
                             Each entry: {gene_id, gene_name, synonyms, gene_products}
documents         list     — PubMed URLs for cited PMIDs
docs              list     — labeled claim–document pairs
query_gene_expansion object — in-text gene detection and benchmark labels (see below)
```

`docs[]` fields: `pmid`, `title`, `abstract_clean`, `year`, `anchor_pos`, `citation_captions`, `doc_match`, `evidence_level`, `reason`

`query_gene_expansion` fields:

| Field | Values | Meaning |
|---|---|---|
| `has_detectable_gene` | yes / no | At least one gene found in query text |
| `n_detected_genes` | int | Number of genes detected |
| `detected_gene_ids` | list | Sorted gene IDs found in query text |
| `detected_gene_expandable` | yes / no | n_det==1 AND that gene has synonyms + products in DB |
| `query_expansion_benchmark` | yes / no | Record is in 7d (see below) |

`genes` uses text-detected records (from `detected_gene_ids`) when any gene is detectable, so it reflects what the query is actually about. The DictyBase curatorial annotation is only kept when nothing is detectable.

### 7d schema

7d is a strict subset of 7a: every record with `query_expansion_benchmark = "yes"` in 7a appears in 7d. A record qualifies when:
1. Exactly one gene is detected in the query text.
2. That gene has non-empty synonyms **and** gene products in DictyBase.
3. Both expansion variants produce a non-empty suffix (i.e. the expansion actually adds new terms not already in the query).

7d drops the `genes` top-level field (redundant with `query_gene_expansion.detected_genes`) and extends `query_gene_expansion` with:

| Field | Meaning |
|---|---|
| `detected_genes` | Full gene records for detected genes |
| `expansion_synonym` | Full expanded query (query + synonym suffix) |
| `expansion_synonym_products` | Full expanded query (query + synonym + product suffix) |

Both expansion fields always have a non-empty suffix for every 7d record — usable for benchmarking either variant.

### Example subsets

`example/dicty_gold_llm_public_train_200.jsonl` and `example/dicty_gold_llm_public_test_50.jsonl` are stratified samples of 7a built with [scripts/public/data_prep/make_goldset_subset.py](../scripts/public/data_prep/make_goldset_subset.py). They include `query_gene_expansion` (with `query_expansion_benchmark`) but omit legacy `query_text_expansion_*` fields. See [example/dicty_gold_llm_public_subset_stats.json](../example/dicty_gold_llm_public_subset_stats.json) for seeds and parameters.

To re-apply expansion to an existing JSONL, use [scripts/public/data_prep/apply_query_expansion.py](../scripts/public/data_prep/apply_query_expansion.py) with config [scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml](../scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml).

### EPMC JSONL (7c)

One JSON object per line from the cleaned EPMC abstracts. Key `abstract` holds the cleaned abstract text.

## Outputs

- `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl` / `.json`
- `output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl` / `.json`
- `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl`
- `output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl` / `.json`

## Summary (short)

We start with Dictybase curator notes and extract claims that cite a dictybase `publication_id`.
Those IDs are mapped to PubMed PMIDs, then joined with EPMC/PubMed title and abstract text.
After cleaning and de-duplication, we label claim-PMID pairs to build a higher-quality goldset
and export the final public JSONs for claims (with labels) and abstracts.



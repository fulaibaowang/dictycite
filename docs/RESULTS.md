# Results and Dataset Notes

This page summarizes dataset preparation outputs and brief evaluation notes.

## Data sources and counts

Curator notes (dictybase):

| Item | Count |
| --- | --- |
| Curator notes (genes) | 1,079 |
| Claims (from curator notes) | 2,063 |

Publication ID to PMID mapping (dictybase):

| Item | Count |
| --- | --- |
| All dictyBase publications with PMID | 4,341 |
| Publications in curator notes with PMID | 1,372 |

Dicty literature from EPMC/PubMed:

| Item | Count |
| --- | --- |
| All Dictyostelium publications on EPMC | 20,447 |
| Claims with titles/abstracts | 2,020 |
| Publications cited by these claims | 1,340 |

## Dataset outputs

Linked sources and cleaned dataset (from notebooks/dastasets.ipynb):

- output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet

## Notes on gold datasets

Using notebooks/dastasets.ipynb, I further curated the dataset:

- Cleaned punctuation noise such as ()[];..
- Grouped claims with pairwise TF-IDF > 0.6
- Targeted about 1,700 claims

For reranker evaluation, two paths are considered:

- Curate gold claim pairs using LLM labels (scripts/public/dicty_claim_labeler.py, notebooks/goldset_llama.ipynb)
- Use BioASQ dataset for a clean benchmark

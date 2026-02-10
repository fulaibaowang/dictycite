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

Linked sources and cleaned dataset (from [notebooks/datasets_merge_clean.ipynb](../notebooks/datasets_merge_clean.ipynb)):

- claim_cleaned_long_pmids_nonNA_abstract.parquet
- gold_with_query_expand.parquet
- articles_all_cleaned_abstract.jsonl

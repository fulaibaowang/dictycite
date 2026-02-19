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

## LLM agreement (3 runs)

Overlap across all three runs: 2,119 claim-PMID pairs.

Doc-match full agreement (pairs / queries):

| Label | Pairs | Queries |
| --- | --- | --- |
| yes | 1,930 | 1,599 |
| no | 154 | 145 |
| unclear | 8 | 8 |

Evidence-level full agreement (pairs / queries):

| Label | Pairs | Queries |
| --- | --- | --- |
| abstract_supports_core | 762 | 654 |
| abstract_supports_detail | 631 | 619 |
| needs_fulltext | 262 | 251 |
| not_applicable | 158 | 150 |

## Public release size

- `dicty_gold_llm_public.json`: 1,656 queries, 2,028 labeled docs, 1,289 unique PMIDs
- `articles_all_cleaned_abstract.jsonl`: 20,447 records

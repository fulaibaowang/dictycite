# DictyCite: data fetching + RAG dataset prep

This repo collects data sources and preprocessing steps for building a Dictyostelium RAG dataset.
It keeps scripts, notebooks, and outputs separate and ready for public release.

## Goal and Plan

- Build a clean claim-citation dataset linking Dictybase curator notes to PubMed metadata.
- Prioritize data quality and traceability over retrieval (retrieval work is not started yet).
- Produce a curated dataset that can be used for downstream RAG and evaluation.

## Dataset Summary (short)

- Claims (from Dictybase curator notes) are linked to dictybase publication IDs.
- Publication IDs are mapped to PMIDs and joined with EPMC titles/abstracts.
- Cleaned output: `output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet`.
- Optional goldset labeling uses LLM review when abstracts are insufficient.

## Docs

- Data prep workflow: [docs/DATA_PREP.md](docs/DATA_PREP.md)
- Detailed commands: [docs/USAGE.md](docs/USAGE.md)
- Results and dataset notes: [docs/RESULTS.md](docs/RESULTS.md)


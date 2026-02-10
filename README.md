# DictyCite: data fetching + RAG dataset prep

This repo collects data sources and preprocessing steps for building a Dictyostelium RAG dataset.
It keeps scripts, notebooks, and outputs.

## Goal and Plan

- Build a clean claim-citation dataset linking Dictybase curator notes to PubMed metadata.
- Build a trustful retrieval system by the dataset (TODO)

## Dataset Summary (short)

- Claims (from Dictybase curator notes) are linked to dictybase publication IDs.
- Publication IDs are mapped to PMIDs and joined with EPMC titles/abstracts.
- Intermediate cleaned datasets are used to build a grouped goldset and LLM labels.
- Public release:
	- `output/cleaned/dicty_gold_llm_public.json` (queries + labeled docs)
	- `output/cleaned/articles_all_cleaned_abstract.json` (EPMC titles/abstracts)

## Docs

- Data prep workflow: [docs/DATA_PREP.md](docs/DATA_PREP.md)
- Detailed commands: [docs/USAGE.md](docs/USAGE.md)
- Results and dataset notes: [docs/RESULTS.md](docs/RESULTS.md)


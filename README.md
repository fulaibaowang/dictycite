# DictyCite: data fetching + RAG dataset prep

This repo collects data sources and preprocessing steps for building a Dictyostelium RAG dataset.
It keeps scripts, notebooks, and outputs.

## Goal and Plan

- Build a clean claim-citation dataset linking Dictybase curator notes to PubMed metadata.
- Build a trustful retrieval system by the dataset (TODO)

## Dataset Summary (short)

- Public release:
	- `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl` (1,656 queries, 2,028 pairs, 1,289 PMIDs)
	- `output/dicty_gold_build/7b_dicty_gold_llm_private.jsonl` (full payload with internal fields)
	- `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl` (20,447 EPMC records)
	- `output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl` (563 queries with both expansion variants)

## Docs

- Data prep workflow: [docs/DATA_PREP.md](docs/DATA_PREP.md)
- Detailed commands: [docs/USAGE.md](docs/USAGE.md)
- Results and dataset notes: [docs/RESULTS.md](docs/RESULTS.md)
- Retrieval pipeline and analysis scripts: [scripts/public/README.md](scripts/public/README.md) (pipeline: [scripts/public/shared_scripts/README.md](scripts/public/shared_scripts/README.md))

## TODO

- colbert

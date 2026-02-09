# Usage

This page collects full command recipes and workflow notes.

## Environment

Set the following environment variables (for example, in .env):

- LLAMA_API_KEY
- NCBI_API_KEY

## Curator notes (dictybase)

Quick test:
```
python scripts/public/dicty_curator_notes.py --limit 10
```

Full dataset:
```
python scripts/public/dicty_curator_notes.py --limit 0 --sleep-base 0.3 --sleep-jitter 0.10
```

Docker example:
```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/dicty_curator_notes.py --limit 10
```

## Publication ID to PMID mapping

Docker example:
```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/dicty_publication.py --limit 10
```

## EPMC full-text fetch

The full fetch workflow is in article_fetching. See:

- article_fetching/README.md

## LLM labeling (goldset)

Command example:
```
python scripts/public/dicty_claim_labeler.py \
  --input_tsv output/cleaned/gold_with_query_expand_flat.tsv \
  --output_jsonl output/llm_labels_goldset_run3.jsonl \
  --sleep_s 0.5 \
  --raise_on_error TRUE
```

Notebook reference:

- notebooks/goldset_llama.ipynb

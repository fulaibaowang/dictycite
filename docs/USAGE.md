# Usage

This page collects full command recipes and workflow notes.

## Curator notes (dictybase)

```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/dicty_curator_notes.py --limit 10
```

## Publication ID to PMID mapping

```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/dicty_publication.py --limit 10
```

## EPMC full-text fetch

See [scripts/public/article_fetching/README.md](../scripts/public/article_fetching/README.md).

## LLM labeling (goldset)

```
python scripts/public/dicty_claim_labeler.py \
  --input_tsv output/cleaned/gold_with_query_expand_flat.tsv \
  --output_jsonl output/llm_labels_goldset_run3.jsonl \
  --sleep_s 0.5 \
  --raise_on_error TRUE
```

Notebook reference:

- [notebooks/goldset_llm_labeling.ipynb](../notebooks/goldset_llm_labeling.ipynb)

# Usage

This page collects full command recipes and workflow notes.
Current work focuses on data preparation; retrieval and modeling workflows may be added later.

## Curator notes (dictybase)

```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/data_prep/dicty_curator_notes.py --limit 10
```

## Publication ID to PMID mapping

```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/data_prep/dicty_publication.py --limit 10
```

## EPMC full-text fetch

See [scripts/public/article_fetching/README.md](../scripts/public/article_fetching/README.md).

## LLM labeling (goldset)

```
python scripts/public/data_prep/dicty_claim_labeler.py \
  --input_tsv output/dicty_gold_build/5b_gold_query_expand_flat.tsv \
  --output_jsonl output/dicty_gold_build/6c_llm_labels_run3.jsonl \
  --sleep_s 0.5 \
  --raise_on_error TRUE
```

Notebook reference:

- [notebooks/goldset_llm_labeling.ipynb](../notebooks/goldset_llm_labeling.ipynb)

## Final public export

Use [notebooks/final_public_export.ipynb](../notebooks/final_public_export.ipynb) to write:

- output/dicty_gold_build/7a_dicty_gold_llm_public.json
- output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl

## Gold-linked full curator notes (build dataset)

From the repository root, after `7a`, `4a`, `1_curator_claims`, `gene_information.txt`, and `2_publication_id_pmid.csv` exist under `output/dicty_gold_build/` (and `dictybase_files/`):

```
.venv/bin/python scripts/public/data_prep/build_gold_linked_notes_dataset.py
```

This re-fetches `summary.json` for every gene linked to the current gold set, writes `8_raw_notes_snapshot.jsonl` and `8a_gold_linked_notes_build.jsonl`, and refreshes `8c` / `8d`. Default behavior **truncates** prior `8_*` artifacts.

Continue an interrupted run (reuse raw snapshots, append only missing genes):

```
.venv/bin/python scripts/public/data_prep/build_gold_linked_notes_dataset.py \
  --resume --resume-build --no-overwrite
```

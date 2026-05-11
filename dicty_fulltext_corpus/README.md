# DictyCite full-text retrieval corpora

Unified `corpus.jsonl` files for retrieval indexing. Each row is a single
indexable document — either a curated abstract row or a body / caption chunk
from a PDF, distinguished by the `type` field.

## Versions

| | v1 | v2 (current) |
|---|---|---|
| Abstract source | `dicty_simulated_data/abstracts/corpus.jsonl` (10,880 PMIDs, **no titles**) | `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl` (20,447 PMIDs, **titles for all**) |
| Chunk source | `output/pdf_extraction/v1/chunks.jsonl` (234 PMIDs, manual-screened) | `output/pdf_extraction/v2/chunks.jsonl` (3,556 PMIDs, Tier-4 auto-screened) |
| Scope | example `train_200 ∪ test_50` only | full PDF archive (`pdfs/` + `manual/`) |
| 7a goldset coverage (PMIDs with abstract row) | 1,251 / 1,289 (38 missing) | **1,289 / 1,289** |
| 7a goldset coverage (PMIDs with at least one chunk) | ~217 / 1,289 | **1,124 / 1,289** |
| Total docs (rows) | 17,368 | 107,475 |
| File | `v1/corpus.jsonl` | `v2/corpus.jsonl` |

v1 is preserved for reproducibility of earlier experiments. New runs should
use v2.

## Row schema

```json
{
  "docno":          "<pmid>#abstract" | "<pmid>#body_001" | "<pmid>#caption_001",
  "pmid":           "<pmid>",
  "title":          "<title>",
  "type":           "abstract" | "body" | "caption",
  "text":           "<text>",
  "seq":            1,            // chunks only
  "n_chars":        1842,         // chunks only
  "position_frac":  0.0           // body chunks only
}
```

`docno` is unique across the whole corpus; `pmid` is the group key for
per-paper max-pool aggregation at evaluation time.

## How to rebuild

See `docs/FULLTEXT.md` for the full pipeline. The v2 corpus was built with:

```bash
python -m scripts.public.pdf_processing.build_corpus \
    --abstracts output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
    --chunks output/pdf_extraction/v2/chunks.jsonl \
    --titles output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
    --out dicty_fulltext_corpus/v2/corpus.jsonl
```

## Companion artifacts

- `output/pdf_extraction/v2/flag_report.jsonl` — per-PDF cleaning outcomes
  (errors, tier-4 flags). The chunker reads this and skips PMIDs flagged
  `low_chars_per_page` (scan signature).
- `output/dicty_gold_build/7a_dicty_gold_pdf_coverage.tsv` — public per-claim
  coverage map for the goldset (which 7a queries have full-text available).

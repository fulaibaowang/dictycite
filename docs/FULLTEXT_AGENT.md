# Full-text handoff (start here)

For coding agents and collaborators: how the PDF → `corpus.jsonl` path works, and what arrives outside git.

Deeper design / coverage numbers: [`FULLTEXT.md`](FULLTEXT.md). Schema of the corpus: [`dicty_fulltext_corpus/README.md`](../dicty_fulltext_corpus/README.md).

## What you should have

| Artifact | Where | Notes |
|---|---|---|
| This repo (clone) | — | Scripts + abstracts in git |
| `pdfs/` (~3,573 `<pmid>.pdf`) | Outside the repo | You already have this dump |
| `manual/` (68 `<pmid>.pdf`) | Outside the repo | **Transferred** — gap fills not in `pdfs/` |
| `output/pdf_extraction/v2/` | In the clone | **Transferred** (~331MB) — cleaned bodies + chunks |
| `dicty_fulltext_corpus/v2/corpus.jsonl` | In the clone | **Transferred** (gitignored; do not commit) |

Place transferred trees at those exact paths:

```text
dicty_fulltext_corpus/v2/corpus.jsonl
output/pdf_extraction/v2/
  body/<pmid>.txt          # contiguous cleaned full text (use this for "original" text)
  references/<pmid>.txt
  chunks.jsonl             # body/caption chunks that feed the corpus
  flag_report.jsonl        # per-PDF clean flags / skips
```

Put `manual/` next to your existing `pdfs/`:

```text
<path>/dictybase_papers/pdfs/     # yours
<path>/dictybase_papers/manual/   # transferred
```

`manual/` is only papers missing from the original dump, added later for goldset coverage. When re-cleaning from PDFs, pass **both** dirs (`manual/` last so it wins on PMID overlap).

**What to use when:**
- Retrieval → `corpus.jsonl` as-is
- Continuous paper text → `output/pdf_extraction/v2/body/<pmid>.txt`
- Rebuild corpus without re-cleaning PDFs → `build_corpus.py` on `v2/chunks.jsonl` + `7c` abstracts
- Full re-parse from PDFs → need `pdfs/` + `manual/` (commands below)

**Do not commit** the v2 corpus, `output/pdf_extraction/v2/`, or PDF trees (copyright). `v1/corpus.jsonl` in git is a smaller legacy subset only.

## Code that builds the corpus

All under `scripts/public/pdf_processing/` (not `scripts/public/article_fetching/` — that path is EPMC/NCBI JSON full text, unrelated).

| Script | Role |
|---|---|
| `fetch_titles.py` | Cache PubMed titles (validation gate) |
| `clean_pdfs.py` | PDF → cleaned `body/` + `references/` + `flag_report.jsonl` |
| `chunk_bodies.py` | bodies → `chunks.jsonl` |
| `build_corpus.py` | abstracts (`7c`) + chunks → `v2/corpus.jsonl` |

Shared knobs / known-bad PMIDs: `config.py`.

Replace `PDF_ROOT` with your `dictybase_papers` path:

```bash
PDF_ROOT=/path/to/dictybase_papers

python -m scripts.public.pdf_processing.fetch_titles \
    --pdfs-dir "$PDF_ROOT/pdfs" \
    --pdfs-dir "$PDF_ROOT/manual" \
    --out output/pdf_extraction/v1/titles.json

python -m scripts.public.pdf_processing.clean_pdfs \
    --pdfs-dir "$PDF_ROOT/pdfs" \
    --pdfs-dir "$PDF_ROOT/manual" \
    --titles output/pdf_extraction/v1/titles.json \
    --out output/pdf_extraction/v2

python -m scripts.public.pdf_processing.chunk_bodies \
    --in output/pdf_extraction/v2 \
    --titles output/pdf_extraction/v1/titles.json \
    --out output/pdf_extraction/v2/chunks.jsonl

python -m scripts.public.pdf_processing.build_corpus \
    --abstracts output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
    --chunks output/pdf_extraction/v2/chunks.jsonl \
    --titles output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
    --out dicty_fulltext_corpus/v2/corpus.jsonl
```

`7c_articles_cleaned_abstract.jsonl` is in the clone. Coverage of which gold claims have a PDF: `output/dicty_gold_build/7a_dicty_gold_pdf_coverage.tsv` (`pdf_source` = `pdfs` | `manual` | empty).

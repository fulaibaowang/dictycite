# Full-Text Extension to the Retrieval Index

> Collaborator / coding-agent handoff (place `manual/` + `output/pdf_extraction/v2/` + `v2/corpus.jsonl`, rebuild commands): [`FULLTEXT_AGENT.md`](FULLTEXT_AGENT.md).

This document describes how full-text PDF content is added alongside the gold
abstracts in the DictyCite retrieval pipeline, what fraction of the public
goldset (`7a_dicty_gold_llm_public.jsonl`) is covered, and the experimental
design used to compare abstract-only vs. abstract-plus-full-text retrieval.

The goal is to extend retrieval beyond curated abstracts by indexing cleaned,
chunked full text from locally-archived PDFs. The hypothesis is that this
helps the `abstract_insufficient` evidence stratum (where the abstract does
not, on its own, support the gold claim) without hurting the
`abstract_supports_core` / `abstract_supports_detail` strata.

## Corpus version

The current full-text corpus is **v2** (assembled from all available PDFs,
auto-screened for extractability, abstracts and titles sourced from
`7c_articles_cleaned_abstract.jsonl`):

| | v1 (legacy) | v2 (current) |
|---|---|---|
| Scope | PMIDs in `train_200 ∪ test_50` only | All PDFs in `pdfs/` + `manual/` |
| Bad-PDF screen | 4 manually-identified PMIDs | Tier-4 `low_chars_per_page` auto-skip |
| Abstract source | `dicty_simulated_data/abstracts/corpus.jsonl` (10,880 PMIDs, **no titles**) | `7c_articles_cleaned_abstract.jsonl` (20,447 PMIDs, **titles for all**) |
| PMIDs chunked | 234 | **3,556** |
| Total chunks | 6,488 | **87,028** |
| Total corpus docs | 17,368 | **107,475** |
| Unified corpus | `dicty_fulltext_corpus/v1/corpus.jsonl` | `dicty_fulltext_corpus/v2/corpus.jsonl` |
| 7a abstract-row coverage | 1,251 / 1,289 | **1,289 / 1,289** |
| 7a chunk coverage | ~217 / 1,289 | **1,124 / 1,289** |

Notes on the v1 → v2 corpus rebuild:

- v1 was assembled from `dicty_simulated_data/abstracts/corpus.jsonl`, which
  has 10,880 PMIDs and *no* titles. 38 of the 1,289 7a goldset PMIDs were
  silently absent from v1's abstract layer, and every chunk row had an empty
  title field.
- v2 switches the abstract+title source to
  `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl` (the same
  source that the non-chunked full-goldset retrieval pipeline already uses
  as `DOCS_JSONL`). All 1,289 7a PMIDs now appear in the corpus with at
  least an abstract row; every row carries a title. PMIDs that lack a usable
  PDF still get retrieved via their abstract — they simply have no body
  chunks.

## PDF corpus

PDFs live in `/Users/yun/Documents/dictybase_papers/` and are named
`<pmid>.pdf`. Two source directories:

| Directory | Origin | Count |
|---|---|---|
| `pdfs/` | DictyBase archival paper dump (PMC and other historical archive sources; tracked in `sources.csv`) | 3,573 |
| `manual/` | Patched-in PDFs for goldset coverage gaps. Includes 7 PDFs manually placed during the v1 experiment, plus 61 auto-fetched from Europe PMC / PMC OA (script: `scripts/public/pdf_processing/fetch_pdfs.py`) | 68 |
| **Total unique** | | **3,580** |

Not every PDF is goldset-relevant; the archive intentionally goes wider than
the gold set so the same source can serve future labeling rounds.

## Pipeline

Three canonical entry points under `scripts/public/pdf_processing/`:

1. **`fetch_titles.py`** — PMID → PubMed title via NCBI esummary, cached as
   `titles.json`. Used for the validation gate during cleaning.
2. **`clean_pdfs.py`** — Extracts and cleans body text. PyMuPDF naive
   `get_text()` → mechanical cleanup (ligatures, dehyphenation, header/footer
   dedup, watermark strip) → structural split (References, Acknowledgments,
   page-1 cover page on Elsevier preproofs) → per-PDF validation
   (`low_chars_per_page`, `no_references_heading`, `title_not_in_body`).
   Writes `body/<pmid>.txt` and `references/<pmid>.txt`.
3. **`chunk_bodies.py`** — Greedy paragraph-accumulator chunker with
   target=1,500 / cap=1,800 / overlap=200 characters (anchored on the
   abstract median of 1,335 chars). Captions, where detectable, are emitted as
   a separate chunk type. Writes `chunks.jsonl` ready for indexing.

A separate script, **`fetch_pdfs.py`**, downloads missing PDFs into
`manual/`. It calls NCBI idconv to get a PMCID, then tries (1) the Europe
PMC PDF render URL, (2) the PMC Open-Access service as fallback. Pre-2000
papers and paywalled material typically fall through (`no_pmcid` or
`not_oa`).

### Bad-PDF screen

A PDF is excluded from chunking when **any** of these hold:

1. PMID is in the static `KNOWN_BAD` set (see `scripts/public/pdf_processing/config.py`):
   three page-image scans (`7771809`, `7813801`, `10192918`) and one Type1-CFF
   font failure (`11032815`). All four are `abstract_supports_*`, so dropping
   them does not bias the abstract-insufficient experiment.
2. Cleaning raised an exception (recorded in `flag_report.jsonl` as `error`).
3. Tier-4 validation flagged `low_chars_per_page` (body < 500 chars/page —
   the page-image-scan signature). In v2 this excludes **81 additional PDFs**
   that the v1 manual screen never reached.

The screen runs entirely off `flag_report.jsonl`. Other Tier-4 flags
(`no_references_heading`, `title_not_in_body`) are recorded but do **not**
disqualify a PDF — these are informational only.

## Coverage of the public goldset (v2)

The current v2 corpus chunks **87.2% of the 1,289 unique PMIDs** in
`7a_dicty_gold_llm_public.jsonl`. Coverage is roughly uniform across evidence
levels:

| evidence_level | unique PMIDs | have_pdf | usable (chunked) | coverage |
|---|---:|---:|---:|---:|
| abstract_insufficient | 307 | 273 | 269 | 87.6% |
| abstract_supports_detail | 620 | 548 | 537 | 86.6% |
| abstract_supports_core | 362 | 324 | 318 | 87.8% |
| **TOTAL** | **1,289** | **1,145** | **1,124** | **87.2%** |

At the claim level (per `(query_id, pmid)` row):

| evidence_level | claims | has_pdf | usable | coverage |
|---|---:|---:|---:|---:|
| abstract_insufficient | 422 | 382 | 378 | 89.6% |
| abstract_supports_detail | 844 | 743 | 732 | 86.7% |
| abstract_supports_core | 762 | 684 | 672 | 88.2% |
| **TOTAL** | **2,028** | **1,809** | **1,782** | **87.9%** |

The drop from `has_pdf` to `usable` (~1.5 pp) is the auto-screen catching
scans and other extraction failures that v1 never tested.

### Public artifact: per-claim coverage table

The full per-claim coverage map is published as
`output/dicty_gold_build/7a_dicty_gold_pdf_coverage.tsv`:

| column | meaning |
|---|---|
| `query_id` | matches `query_id` in `7a_dicty_gold_llm_public.jsonl` |
| `pmid` | gold doc PubMed ID |
| `evidence_level` | `abstract_insufficient` / `abstract_supports_core` / `abstract_supports_detail` |
| `has_pdf` | `yes` if `<pmid>.pdf` exists in `pdfs/` ∪ `manual/` |
| `pdf_source` | `pdfs`, `manual`, or empty |
| `pdf_usable` | `yes` if the PDF passed the bad-PDF screen (= eligible for chunking) |
| `usable_reason` | empty if `pdf_usable=yes`, else one of: `no_pdf` / `known_bad` / `low_chars_per_page` / `error:...` |
| `in_chunks_v2` | `yes` if the PMID actually appears in `output/pdf_extraction/v2/chunks.jsonl` |

`pdf_usable` and `in_chunks_v2` should agree on every row. This file is the
canonical record of which claims are **eligible** for the full-text
experiment (`pdf_usable=yes`) and which remain abstract-only.

## Experiment design

Two experimental arms:

- **Baseline.** Index contains gold abstracts only (corpus.jsonl).
- **Full-text.** Index contains gold abstracts **plus** body chunks from
  `chunks.jsonl` for every PMID with a PDF (i.e. the ~1,145 PMIDs above).
  Per-PMID max-pooling is applied during evaluation to aggregate scores
  across an article's chunks back to a single document-level score.

Both arms are evaluated against the full set of public goldset queries.
Two reporting views:

1. **Full-cohort metrics, stratified by evidence_level** — the existing
   figure shape (`fig3_candidate_evidence_level_mrr.png`). Reflects what a
   downstream user would observe when querying the index in production:
   queries whose gold PMID has no PDF can move only through index-wide
   side-effects (BM25 IDF shifts, dense space density changes).
2. **Has-PDF subset metrics, stratified by evidence_level** — same two
   arms, but the evaluation set is restricted to queries where
   `has_pdf == yes` in the coverage TSV above. This isolates the causal
   effect of adding full-text for queries that can directly benefit from
   it, and is the cleaner number for "did the technique work?"

Both views are reported. The full-cohort view is the production-reality
headline; the has-PDF view is the technique-effect estimate.

## Reproducing the corpus

```bash
# 1. Cache PubMed titles (used for validation only)
python -m scripts.public.pdf_processing.fetch_titles \\
    --pdfs-dir /Users/yun/Documents/dictybase_papers/pdfs \\
    --pdfs-dir /Users/yun/Documents/dictybase_papers/manual \\
    --out output/pdf_extraction/v1/titles.json

# 2. (Optional) Fetch any missing PDFs from PMC/EPMC into manual/
python -m scripts.public.pdf_processing.fetch_pdfs \\
    --pmids-file output/pdf_extraction/missing_pmids_7a.txt \\
    --out-dir /Users/yun/Documents/dictybase_papers/manual \\
    --skip-dir /Users/yun/Documents/dictybase_papers/pdfs \\
    --skip-dir /Users/yun/Documents/dictybase_papers/manual \\
    --report output/pdf_extraction/fetch_pdfs_report.json

# 3. Clean PDFs (all of pdfs/ + manual/, scans auto-flagged by Tier 4)
python -m scripts.public.pdf_processing.clean_pdfs \\
    --pdfs-dir /Users/yun/Documents/dictybase_papers/pdfs \\
    --pdfs-dir /Users/yun/Documents/dictybase_papers/manual \\
    --titles output/pdf_extraction/v1/titles.json \\
    --out output/pdf_extraction/v2

# 4. Chunk the cleaned bodies (skips errors + Tier 4 low_chars_per_page)
python -m scripts.public.pdf_processing.chunk_bodies \\
    --in output/pdf_extraction/v2 \\
    --titles output/pdf_extraction/v1/titles.json \\
    --out output/pdf_extraction/v2/chunks.jsonl

# 5. Build the unified retrieval corpus (abstracts + chunks)
#    Use 7c as the abstracts source — it has titles for all rows and full
#    coverage of the 7a goldset (the legacy abstracts/corpus.jsonl was
#    missing 38 goldset PMIDs and had no titles).
python -m scripts.public.pdf_processing.build_corpus \\
    --abstracts output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \\
    --chunks output/pdf_extraction/v2/chunks.jsonl \\
    --titles output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \\
    --out dicty_fulltext_corpus/v2/corpus.jsonl
```

Pass `--pmids <file>` to step 3 to restrict to a subset (e.g. for a smaller
ablation). The v1 experiment used `train_200 ∪ test_50` (234 PDFs); the v2
default is "all PDFs in `pdfs/` + `manual/` minus auto-flagged scans"
(3,556 PDFs).

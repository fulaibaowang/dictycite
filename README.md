# DictyCite: a *Dictyostelium discoideum* claim–citation dataset and retrieval benchmark for the Dictyostelium domain

DictyCite is a public **claim–citation goldset** built from dictyBase curator notes (1,656 queries / 2,028 claim–PMID pairs / 1,289 PMIDs) and a paired **two-stage retrieval pipeline** (BM25 + dense → cross-encoder rerank) evaluated against it. The dataset, the cleaned EPMC abstract corpus (20,447 records), a gene-aware query-expansion benchmark subset (563 queries), and stratified train/test splits are released alongside the code that builds them and the notebooks that produce every figure and table in the accompanying paper.

## Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) <!-- TODO: replace with real DOI after Zenodo release -->

| File | Records | Description |
|---|---|---|
| [`output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl`](output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl) | 1,656 queries / 2,028 pairs / 1,289 PMIDs | Public goldset: curator claim + cited PMIDs + LLM evidence labels |
| [`output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl`](output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl) | 20,447 | Cleaned EPMC abstracts (paired corpus for retrieval) |
| [`output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl`](output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl) | 563 | Gene-aware query-expansion benchmark subset (both expansion variants) |
| [`example/dicty_gold_llm_public_train_200.jsonl`](example/dicty_gold_llm_public_train_200.jsonl) / [`example/dicty_gold_llm_public_test_50.jsonl`](example/dicty_gold_llm_public_test_50.jsonl) | 200 / 50 | Stratified train / test sample of the public goldset |

Full schema: [`docs/DATA_PREP.md`](docs/DATA_PREP.md) §6.

---

## Part A — Repository

### Running the pipeline

#### Build the goldset
```bash
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 \
  fulaibaowang/dictycite:16.01.2026 \
  python /dictycite/scripts/public/data_prep/dicty_curator_notes.py --limit 10
```
Full workflow (curator notes → publication ID mapping → cleaning → LLM labelling → public export): [`docs/DATA_PREP.md`](docs/DATA_PREP.md). Command examples for every stage: [`docs/USAGE.md`](docs/USAGE.md).

#### Fetch articles from Europe PMC / PubMed
```bash
docker run --rm -v "$(pwd)/scripts/public/article_fetching/output:/app/output" \
  fulaibaowang/dictyfetch:19.01.2026 \
  --query "Dictyostelium discoideum" --output_path /app/output
```
License filtering, full-text retrieval modes, and parser options: [`scripts/public/article_fetching/README.md`](scripts/public/article_fetching/README.md).

#### Retrieval + reranking
```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh \
  --config scripts/public/shared_scripts/conf/workflow_config_document.env
```
The retrieval stack (BM25 + dense → RRF fusion → cross-encoder rerank → post-rerank fusion, with optional snippet-RRF) is a git subtree of the standalone [`RAG-scripts`](https://github.com/fulaibaowang/RAG-scripts) project; the pipeline flowchart and full parameter reference live in that repo's README.

#### Ragnarok baseline (external reference, §4.4)
BM25 + RankZephyr listwise rerank using the upstream Ragnarok / RankLLM stack; image and runner under [`scripts/public/ragnarok_baseline/`](scripts/public/ragnarok_baseline/).

#### Full-text / chunked-corpus variant (§4.3)
Indexing cleaned, chunked PDFs alongside abstracts (v2 corpus: 3,556 PMIDs / 87,028 chunks). Coverage, build steps, and the abstract-only-vs-chunked experimental design: [`docs/FULLTEXT.md`](docs/FULLTEXT.md).

### Repo layout

| Directory | What it is |
|---|---|
| [`scripts/public/`](scripts/public/) | Data preparation, article fetching, PDF processing, Ragnarok baseline, and the shared retrieval pipeline |
| [`notebooks/`](notebooks/) | Interactive workflows (`01_*`–`07_*`) and the figure-generating notebooks for the paper |
| [`docs/`](docs/) | Workflow, schema, methods, results, and full-text extension documentation |
| [`output/dicty_gold_build/`](output/dicty_gold_build/) | Released goldset JSONLs, expansion benchmark, EPMC abstracts |
| [`output/paper_figures/`](output/paper_figures/) | Final figures and tables in the paper |
| [`example/`](example/) | Stratified train / test sample splits |
| [`dictybase_files/`](dictybase_files/) | Upstream dictyBase metadata (gene IDs, synonyms, products) |
| [`dicty_fulltext_corpus/`](dicty_fulltext_corpus/) | Chunked full-text corpus (v2; rebuild via `scripts/public/pdf_processing/build_corpus.py`) |

### Docs

- [`docs/DATA_PREP.md`](docs/DATA_PREP.md) — end-to-end data preparation workflow and goldset schema
- [`docs/USAGE.md`](docs/USAGE.md) — command recipes for each pipeline stage
- [`docs/METHODS.md`](docs/METHODS.md) — paper-ready methods description of goldset construction
- [`docs/RESULTS.md`](docs/RESULTS.md) — dataset statistics and LLM-labelling agreement
- [`docs/FULLTEXT.md`](docs/FULLTEXT.md) — full-text PDF corpus, coverage, and abstract-vs-chunked experimental design
- [`scripts/public/shared_scripts/README.md`](scripts/public/shared_scripts/README.md) — retrieval pipeline quickstart and parameter reference

### License and acknowledgements

Released under the [Apache License 2.0](LICENSE).

Upstream data sources: dictyBase (curator notes, gene metadata, publication references), Europe PMC and PubMed (article metadata and abstracts). The LLM-validated label layer on the public goldset is produced by three independent Llama 3.3 labelling runs with a full-agreement filter; full provenance is documented in [`docs/METHODS.md`](docs/METHODS.md).

---

## Part B — Paper

Notebooks are paired with `.py` files via jupytext (`ipynb,py:percent`); edit either and run `jupytext --sync <file>` to update the partner.

### Main figures

| Figure | Notebook |
|---|---|
| Fig 3 — main ranker comparison | [`notebooks/ragnarok_comparison.ipynb`](notebooks/ragnarok_comparison.ipynb) |
| Fig 4 — gene-aware query expansion | [`notebooks/query_expansion_sweeping.ipynb`](notebooks/query_expansion_sweeping.ipynb) |
| Fig 5 — evidence-level retrieval / rerank, has-PDF subset | [`notebooks/report_7a.ipynb`](notebooks/report_7a.ipynb) |

### Supplementary figures

Grouped by source notebook:

- [`notebooks/report_7a.ipynb`](notebooks/report_7a.ipynb) — **Fig S1** (K calibration), **Fig S5** (evidence-level, all claims), **Fig S6** (per-query chunked Δ, has-PDF)
- [`notebooks/ragnarok_comparison.ipynb`](notebooks/ragnarok_comparison.ipynb) — **Fig S2** (full ranker comparison), **Fig S3** (per-query rerank Δ)
- [`notebooks/query_expansion_sweeping.ipynb`](notebooks/query_expansion_sweeping.ipynb) — **Fig S4** (QE × reranker MRR panels)

### Tables

Tables 1–3 and Table S1 are hand-curated `.md` files under [`output/paper_figures/`](output/paper_figures/); the underlying bootstrap-CI and summary CSVs are written by the same three notebooks above (e.g. `mrr10_bootstrap_ci_vs_bm25.csv`, `qe_mrr10_bootstrap_ci_vs_body.csv`).

All artifacts under `output/paper_figures/` are final.

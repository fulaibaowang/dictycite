# DictyCite: a *Dictyostelium discoideum* claim–citation dataset and retrieval benchmark for the Dictyostelium domain

DictyCite is a public *Dictyostelium discoideum* claim–citation goldset paired with a two-stage retrieval pipeline (BM25 + dense → cross-encoder rerank) evaluated against it. The dataset is built from dictyBase curator notes and joined with cleaned Europe PMC abstracts. The repository contains the data preparation code, the retrieval pipeline, and the notebooks that produce all paper figures.

## Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) <!-- TODO: replace with real DOI -->

| File | Records | Description |
|---|---|---|
| [`7a_dicty_gold_llm_public.jsonl`](output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl) | 1,656 queries / 2,028 pairs | Curator claim + cited PMIDs + LLM evidence labels |
| [`7c_articles_cleaned_abstract.jsonl`](output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl) | 20,447 articles | Cleaned EPMC abstract corpus |
| [`7d_dicty_gold_query_expansion_benchmark.jsonl`](output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl) | 563 queries | Gene-aware query-expansion benchmark subset |
| [`example/`](example/) | 200 / 50 | Stratified train / test sample |

Full schema: [`DATA_PREP.md`](docs/DATA_PREP.md).

## Pipeline

- **Build the goldset** — curator notes → publication ID mapping → cleaning → LLM labelling → public export. See [`DATA_PREP.md`](docs/DATA_PREP.md) and [`USAGE.md`](docs/USAGE.md).
- **Fetch articles** — Europe PMC / PubMed metadata and full text. See [`article_fetching/`](scripts/public/article_fetching/README.md).
- **Retrieval and reranking** — BM25 + dense → RRF fusion → cross-encoder rerank → post-rerank fusion. The stack is a git subtree of the standalone [`RAG-scripts`](https://github.com/fulaibaowang/RAG-scripts) project; its README has the pipeline flowchart and parameter reference.
- **Full-text / chunked corpus** — indexing chunked PDFs alongside abstracts. See [`FULLTEXT.md`](docs/FULLTEXT.md).
- **Ragnarok baseline** — external reference pipeline (BM25 + RankZephyr listwise rerank). See [`ragnarok_baseline/`](scripts/public/ragnarok_baseline/).

## Paper figures

Three notebooks produce all paper figures into [`output/paper_figures/Figures/`](output/paper_figures/Figures/) (Overleaf-aligned); the hand-curated tables under [`output/paper_figures/`](output/paper_figures/) are fed by bootstrap-CI and summary CSVs the same notebooks write. Manuscript figure numbers differ from file names — see [`FIGURES.md`](output/paper_figures/FIGURES.md) for the full figure/table index and underlying-data paths. Fig 1 and Fig 2 are external (hand-drawn) and not produced by these notebooks.

- [`ragnarok_comparison.ipynb`](notebooks/ragnarok_comparison.ipynb) — Fig S2 (and the ranker-comparison plot now folded into Table 1)
- [`query_expansion_sweeping.ipynb`](notebooks/query_expansion_sweeping.ipynb) — Fig 3
- [`report_7a.ipynb`](notebooks/report_7a.ipynb) — Fig 4, Fig S1 (plus diagnostic Fig S3, Fig S4)

## Repository layout

| Directory | Contents |
|---|---|
| [`scripts/`](scripts/) | Data preparation, article fetching, PDF processing, Ragnarok baseline, retrieval pipeline |
| [`notebooks/`](notebooks/) | Interactive workflows and figure-generating notebooks |
| [`docs/`](docs/) | Workflow, schema, methods, results, and full-text documentation |
| [`output/`](output/) | Released goldset, paper figures, derived artifacts |
| [`example/`](example/) | Stratified train / test sample splits |
| [`dictybase_files/`](dictybase_files/) | Upstream dictyBase metadata |
| [`dicty_fulltext_corpus/`](dicty_fulltext_corpus/) | Chunked full-text corpus |

## Documentation

- [`DATA_PREP.md`](docs/DATA_PREP.md) — data preparation workflow and goldset schema
- [`USAGE.md`](docs/USAGE.md) — command recipes for each stage
- [`METHODS.md`](docs/METHODS.md) — goldset construction methods
- [`RESULTS.md`](docs/RESULTS.md) — dataset statistics and labelling agreement
- [`FULLTEXT.md`](docs/FULLTEXT.md) — full-text PDF corpus and chunked retrieval

## License and acknowledgements

Released under the [Apache License 2.0](LICENSE). Upstream data sources: dictyBase, Europe PMC, and PubMed.

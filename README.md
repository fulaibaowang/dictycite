# DictyCite: a *Dictyostelium discoideum* claim–citation dataset and retrieval benchmark for the Dictyostelium domain

DictyCite is a public *Dictyostelium discoideum* claim–citation goldset paired with a two-stage retrieval pipeline (BM25 + dense → cross-encoder rerank) evaluated against it. The dataset is built from dictyBase curator notes and joined with cleaned Europe PMC abstracts. The repository contains the data preparation code, the retrieval pipeline, and the notebooks that produce all paper figures.

## Paper

This repository accompanies our Discovery Science 2026 paper:

> Yun Wang, Gad Shaulsky, Tomaž Curk, Blaž Zupan. *Benchmarking Literature Retrieval for a Model Organism: A* Dictyostelium *Case Study.* Discovery Science (DS 2026), Lecture Notes in Computer Science, Springer, 2026 (to appear). \
> **[[PDF]](output/paper_figures/dictycite_ds2026_accepted_manuscript.pdf)** (accepted manuscript)

```bibtex
@inproceedings{wang2026dictycite,
  title     = {Benchmarking Literature Retrieval for a Model Organism: A \emph{Dictyostelium} Case Study},
  author    = {Wang, Yun and Shaulsky, Gad and Curk, Toma{\v{z}} and Zupan, Bla{\v{z}}},
  booktitle = {Discovery Science: 29th International Conference, DS 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

The paper corresponds to tag [**`v0.3.0`**](https://github.com/fulaibaowang/dictycite/tree/v0.3.0). The vendored pipeline is frozen
at that state; for a maintained version, use [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts).

## Dataset

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20308282-blue)](https://doi.org/10.5281/zenodo.20308282)

| File | Records | Description |
|---|---|---|
| [`7a_dicty_gold_llm_public.jsonl`](output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl) | 1,656 queries / 2,028 pairs | Curator claim + cited PMIDs + LLM evidence labels |
| [`7c_articles_cleaned_abstract.jsonl`](output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl) | 20,447 articles | Cleaned EPMC abstract corpus |
| [`7d_dicty_gold_query_expansion_benchmark.jsonl`](output/dicty_gold_build/7d_dicty_gold_query_expansion_benchmark.jsonl) | 563 queries | Gene-aware query-expansion benchmark subset |
| [`example/`](example/) | 200 / 50 | Stratified train / test sample |

Full schema: [`DATA_PREP.md`](docs/DATA_PREP.md).

## Pipeline

- **Build the goldset** — curator notes → publication ID mapping → cleaning → LLM labelling → public export. See [`DATA_PREP.md`](docs/DATA_PREP.md) and [`USAGE.md`](docs/USAGE.md). The evidence-level labelling prompt is in [`dicty_claim_labeler.py`](scripts/public/data_prep/dicty_claim_labeler.py).
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
- [`AGENTS.md`](AGENTS.md) — short orientation: what the repo is, what stays fixed, where to look

## License and acknowledgements

Released under the [Apache License 2.0](LICENSE). Upstream data sources: dictyBase, Europe PMC, and PubMed.

The paper PDF (`output/paper_figures/dictycite_ds2026_accepted_manuscript.pdf`) is the authors' accepted manuscript. It is not covered by the Apache License. Its use is subject to the [Springer Nature Accepted Manuscript terms of use](https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms).

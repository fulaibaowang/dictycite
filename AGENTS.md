# AGENTS.md

A short orientation to this repository, for people and coding agents alike. The details live in the
docs linked below. This page says what the repo is and where to look.

## What this is

The code and data behind a Discovery Science 2026 paper that benchmarks literature retrieval for
*Dictyostelium discoideum*. It is a finished study, not a maintained tool:

- **Benchmark.** Claims written by dictyBase curators, linked to the PubMed articles they cite, with
  LLM evidence labels. A Europe PMC abstract corpus comes with it. Released on Zenodo
  (see [README](README.md#dataset)).
- **Experiments.** A retrieval and reranking pipeline run over the benchmark. It compares rerankers,
  tests gene-aware query expansion, and compares abstract-only with full-text retrieval.
- **Paper.** Notebooks that produce the figures and tables, and the LaTeX source.

The pipeline is [RAG-scripts](https://github.com/fulaibaowang/RAG-scripts), vendored under
`scripts/public/shared_scripts/`. To build your own system, use RAG-scripts directly.

## What stays fixed

- **The paper is tag `v0.3.0`.** The vendored pipeline is frozen at that state. RAG-scripts itself
  is still developed, so pipeline changes belong there.
- **The released files match the Zenodo DOI.** Those are `7a`, `7c` and `7d` under
  `output/dicty_gold_build/`, plus `example/`. Don't regenerate them in place. dictyBase and Europe
  PMC change over time and the LLM labels are not deterministic, so re-running the build drifts.
  Reproduction starts from the released files.

## Where to look

| To find out | Read |
|---|---|
| How the benchmark was built (numbered stages in `output/dicty_gold_build/`) | [docs/DATA_PREP.md](docs/DATA_PREP.md), [docs/METHODS.md](docs/METHODS.md) |
| What each dataset field means | [docs/DATA_PREP.md](docs/DATA_PREP.md), [zenodo_release/README.md](zenodo_release/README.md) |
| Dataset counts and labelling agreement | [docs/RESULTS.md](docs/RESULTS.md) |
| Commands for each data step | [docs/USAGE.md](docs/USAGE.md) |
| Which notebook makes which figure or table | [output/paper_figures/FIGURES.md](output/paper_figures/FIGURES.md) (paper numbers differ from file names) |
| The full-text corpus | [docs/FULLTEXT.md](docs/FULLTEXT.md) |
| How to run the pipeline, and what each parameter does | [scripts/public/shared_scripts/README.md](scripts/public/shared_scripts/README.md) |

The configs that ran the paper's experiments are under `scripts/private_scripts/hpc_scripts/`. They
were written for SLURM clusters and won't run elsewhere as they are. If you reuse one anyway, two
settings in them are misleading:

- **Retrieval fusion.** The paper uses RRF with k=60 and equal BM25/dense weights. The frida configs
  set `RETRIEVAL_FUSION_K_RRF=150`, but the paper runs never used it: they ran an earlier fusion
  step that swept k ∈ {60, 100} and three weightings and kept the best — k=60 with equal weights
  (two variants in the 7d query-field sweep picked 1:2). The frozen pipeline applies the configured
  k directly, so set 60 to match the paper.
- **The `vega/` configs use pre-rename `HYBRID_*` variables**, which the pipeline silently ignores.

For a run's exact settings, trust the record the run writes (for example
`retrieval/fusion/best_config.json`) over the config that launched it.

## Editing

- Notebooks are paired with jupytext. Edit the `.py` file, then run
  `jupytext --sync notebooks/<name>.py`.
- Match the surrounding code. There is no linter config.

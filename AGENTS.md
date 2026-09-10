# AGENTS.md

A guide to this repository, for people and coding agents alike: what was done, where each result in
the paper comes from, and what should stay fixed. To use the dataset, start with
[README.md](README.md). To build your own retrieval system, use
[RAG-scripts](https://github.com/fulaibaowang/RAG-scripts), which is the maintained pipeline. This repo
is the record of one study that used it.

## What this is

The code and data behind a Discovery Science 2026 paper that benchmarks literature retrieval for
*Dictyostelium discoideum*, a model organism. It holds three things:

- **The benchmark.** Curator-written claims from dictyBase, the PubMed articles each claim cites, and
  LLM-assigned evidence labels, over a Europe PMC abstract corpus. Released on
  [Zenodo](https://doi.org/10.5281/zenodo.20308282).
- **The experiments.** Runs of a retrieval and reranking pipeline over that benchmark. Three questions
  were studied: which cross-encoder reranker to use, whether gene-aware query expansion helps, and
  abstract-only versus full-text retrieval.
- **The paper.** Notebooks that turn run outputs into the figures and tables, plus the LaTeX source.

It is a finished study, not a maintained tool. Most of the work was data curation and analysis. The
only substantial software is the vendored pipeline, and that lives upstream.

## Which version is the paper

| Tag | State |
|---|---|
| `v0.1.0` | Early pre-submission draft |
| `v0.2.0` | Version submitted to Discovery Science 2026 |
| **`v0.3.0`** | **Camera-ready: the published paper** |

`main` may move past `v0.3.0` for documentation. For anything that has to match the paper, check out
`v0.3.0`.

The pipeline under `scripts/public/shared_scripts/` is a git subtree of RAG-scripts. At `v0.3.0` it
is identical to RAG-scripts commit
[`40793a5`](https://github.com/fulaibaowang/RAG-scripts/commit/40793a5) (2026-07-08). That sync came
after submission, but it only added generation features, CI and opt-in first-stage modes. The
retrieval, fusion and reranking code, and every default, are the same as at `v0.2.0`.

**The vendored copy and the released data are frozen here.** RAG-scripts itself is still actively
developed and has moved on since `40793a5`.
Read it for reference, but don't assume its current features exist here, and don't run
`git subtree pull` to refresh it. The released files (below) are the ones the Zenodo DOI points to.

## How the benchmark was built

Every build artifact lives under `output/dicty_gold_build/`, and its number prefix is its stage:

| # | Step | Code |
|---|---|---|
| 1 | Download dictyBase curator notes, extract claims with citation anchors | `data_prep/dicty_curator_notes.py`, notebook `01` |
| 2 | Map dictyBase publication ids to PMIDs | `data_prep/dicty_publication.py`, notebook `02` |
| 3 | Fetch titles and abstracts from Europe PMC | `article_fetching/`, notebook `03` |
| 4 | Merge, clean, deduplicate, and group near-duplicate claims | notebook `04` |
| 5 | Gene-aware query expansion from dictyBase synonyms and products | notebook `05`, `data_prep/apply_query_expansion.py` |
| 6 | LLM evidence labels for each claim–article pair, three runs for agreement | `data_prep/dicty_claim_labeler.py`, notebook `06` |
| 7 | Public export: goldset `7a`, abstract corpus `7c`, query-expansion subset `7d` | notebook `07` |
| 8 | Full curator notes linked to the goldset (supporting material) | `data_prep/build_gold_linked_notes_dataset.py` |

(`data_prep/` and `article_fetching/` are under `scripts/public/`, notebooks are under `notebooks/`.)
The step-by-step workflow and schemas are in [docs/DATA_PREP.md](docs/DATA_PREP.md). The method in
paper form, followed by a technical reference, is in [docs/METHODS.md](docs/METHODS.md).

Re-running steps 1–3 today will not give the same data. dictyBase and Europe PMC both change over
time, and the LLM labels in step 6 are not deterministic. That is why the labels (`6a`–`6c`) and
the released files are committed. **Reproducing the study starts from `7a` and `7c`, not from the
scrape.**

### Released files: don't regenerate them

`7a_dicty_gold_llm_public.jsonl`, `7c_articles_cleaned_abstract.jsonl`,
`7d_dicty_gold_query_expansion_benchmark.jsonl` and the `example/` splits are the benchmark. The
`zenodo_release/` files are the same data under public names. Re-running notebooks `04`–`07` rewrites
the build files in place. If the result differs at all, it silently stops matching the paper and
the DOI. Write any new variant under a new filename.

## How the experiments were run

Each experiment is one env config passed to the vendored orchestrator:

```bash
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config my_run.env
```

The pipeline runs BM25 and dense retrieval, fuses them with RRF, reranks with a cross-encoder, then
fuses the reranked and first-stage rankings again. The configs behind the paper are
the files named `config_*.env` under `scripts/private_scripts/hpc_scripts/`. They were written for
two SLURM clusters (`frida/` and `vega/`), so the paths won't work elsewhere. Their names tell you
which experiment each one ran. For the exact settings, trust the record a run writes over the config
that launched it. Each stage saves its resolved arguments (for example `retrieval/fusion/config.json`
and `best_config.json`). Retrieval fusion, for instance, was run as a small sweep
(k ∈ {60, 100} × three BM25:dense weightings), and the paper reports the chosen setting.

| Config name contains | Experiment |
|---|---|
| `7a_public_goldset` (plus `_rerank_<model>`) | Ranker comparison on the full goldset |
| `query_field_sweep_7d` | Query expansion on the 563-query subset |
| `chunked_fulltext_v2` | Abstracts plus full-text chunks |

The external baseline (BM25 followed by RankZephyr listwise reranking) is in
`scripts/public/ragnarok_baseline/`.

To run something yourself, start from `scripts/public/shared_scripts/conf/workflow_config_document.env`
(the document route, which is what the paper uses).
Use `7a` as the queries and build indexes from `7c`, or from the full-text corpus. Index building and
every parameter are documented in the
[shared_scripts README](scripts/public/shared_scripts/README.md). Reranking needs a GPU. The rest
runs on a workstation.

**Full text** is the one part a clone can't rebuild. The PDFs are copyrighted and not distributed,
and the chunked corpus built from them (`dicty_fulltext_corpus/v2/`) is too large for git. See
[docs/FULLTEXT.md](docs/FULLTEXT.md) for the design and coverage, and
[docs/FULLTEXT_AGENT.md](docs/FULLTEXT_AGENT.md) for rebuilding the corpus from a PDF archive.
`7a_dicty_gold_pdf_coverage.tsv` lists which cited articles have full text.

## From runs to the paper

[output/paper_figures/FIGURES.md](output/paper_figures/FIGURES.md) maps every figure and table in the
manuscript to the notebook that makes it and the data it reads. **Manuscript figure numbers differ
from file names**, and that file is the key between them. Three notebooks do the work:
`ragnarok_comparison`, `query_expansion_sweeping` and `report_7a`. They read run outputs under
`output/workflow_*`, which are not committed. So regenerating a figure means running the pipeline
first. The resulting tables (`output/paper_figures/table*.md`, `.csv`) and PNGs are committed.

## What is not in git

| Missing | Why | How to get it |
|---|---|---|
| Retrieval indexes (`indexes/`) | Large, derived | Build from `7c` or the full-text corpus |
| Run outputs (`output/workflow_*`) | Large, derived | Run the pipeline |
| PDFs, `output/pdf_extraction/v2/`, `dicty_fulltext_corpus/v2/corpus.jsonl` | Copyright and size | [docs/FULLTEXT_AGENT.md](docs/FULLTEXT_AGENT.md) |
| Build stages 1–5, `7b` (private payload) | Intermediate or internal | Re-run the notebooks (results will drift, see above) |
| API keys | Secrets | `.env` with `LLAMA_API_KEY`, `NCBI_API_KEY` |

## If you change something

- **Pipeline code changes go to RAG-scripts.** Retrieval, fusion, reranking and generation live there.
  Changing the vendored copy here would break its match with `40793a5`.
- **Notebooks are paired with jupytext** (`ipynb,py:percent`, see `pyproject.toml`). The `.py` is the
  one to edit and review. Run `jupytext --sync notebooks/<name>.py` afterwards.
- **Don't overwrite released files.** Write new variants under new names (see above).
- **Match the file you are editing.** The code is argparse CLIs, Polars for tables, and JSONL for
  anything record-shaped. There is no linter or formatter config.

## Where the answers are

| Question | Source |
|---|---|
| What is in each dataset file, field by field? | [docs/DATA_PREP.md](docs/DATA_PREP.md), [zenodo_release/README.md](zenodo_release/README.md) |
| How exactly was the goldset constructed? | [docs/METHODS.md](docs/METHODS.md) |
| Dataset counts and labelling agreement | [docs/RESULTS.md](docs/RESULTS.md) |
| Which notebook makes which figure or table? | [output/paper_figures/FIGURES.md](output/paper_figures/FIGURES.md) |
| Command recipes for the data steps | [docs/USAGE.md](docs/USAGE.md) |
| What does a pipeline knob do? | [shared_scripts README](scripts/public/shared_scripts/README.md), `run_retrieval_rerank_pipeline.sh --help` |
| Full text: design, coverage, rebuild | [docs/FULLTEXT.md](docs/FULLTEXT.md), [docs/FULLTEXT_AGENT.md](docs/FULLTEXT_AGENT.md) |

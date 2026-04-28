# scripts/public

| Subdirectory / file | What it is |
|---|---|
| [`data_prep/`](data_prep/README.md) | Build the gold dataset (curator notes → publication map → LLM labels → public export) |
| [`article_fetching/`](article_fetching/README.md) | Fetch article metadata + full text from Europe PMC / NCBI |
| [`shared_scripts/`](shared_scripts/README.md) | Generic BM25 + dense + reranker retrieval pipeline (git subtree) |
| `plot_by_evidence_level.py` | Stratified recall / MAP plots by `evidence_level` from gold JSON |

The notebooks in `notebooks/` (`01_*` … `07_*`) are the interactive counterparts of `data_prep/`; either entry point produces the same artifacts.

## plot_by_evidence_level.py

Generates three figures from a workflow output directory and gold JSON: hybrid recall curve, rerank recall, and rerank MAP@10 — each with one subplot per `evidence_level`. Saves under `workflow_dir/{hybrid,rerank}/figures/`. Run with `--help` for arguments.

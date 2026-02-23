# Public scripts

Scripts in this directory support the retrieval pipeline and analysis. The main pipeline (BM25 → Dense → Hybrid → Reranker) and its configuration are documented in **[shared_scripts/README.md](shared_scripts/README.md)**.

## Analysis and comparison scripts

### combine_query_field_sweep_results.py

Combines metrics from a 3×3 query-field sweep (BM25 × Dense, optional Hybrid). Reads `bm25_*_dense_*/{bm25,dense}/metrics.csv` and optionally `hybrid/results_all.csv`, adds combo and query-field labels, writes one combined CSV with train/test preserved, and plots recall curves (one figure per batch).

**Usage:**

```bash
python scripts/public/combine_query_field_sweep_results.py
python scripts/public/combine_query_field_sweep_results.py --workflow_dir output/workflow_hpc_test --no_plot
python scripts/public/combine_query_field_sweep_results.py --plot bm25 dense   # default
python scripts/public/combine_query_field_sweep_results.py --plot hybrid
python scripts/public/combine_query_field_sweep_results.py --plot bm25 dense hybrid
python scripts/public/combine_query_field_sweep_results.py --log_x   # log-scale x-axis (K)
python scripts/public/combine_query_field_sweep_results.py --log_y   # log-scale y-axis (recall)
```

### plot_by_evidence_level.py

Generates pipeline-style plots stratified by **evidence_level** from the gold JSON: (1) hybrid recall curve with one subplot per evidence level; (2) rerank recall figure by evidence level; (3) rerank MAP@10 figure by evidence level. Saves to `workflow_dir/hybrid/figures/` and `workflow_dir/rerank/figures/` (or `--rerank-dir`).

**Usage:** Run from repo root with paths to the workflow output and gold JSON(s). See script docstring and `--help` for required arguments.

### compare_result_dirs.py (shared_scripts)

Compares metrics and optionally plots recall and/or MAP curves across two or more result directories (e.g. rerank_body vs rerank_synonyms vs rerank_long). Supports dirs with `metrics.csv` or dirs with only `runs/*.tsv` (e.g. hybrid) when `--train-json` and/or `--test-batch-jsons` are provided. See [shared_scripts/README.md](shared_scripts/README.md) and script `--help`.

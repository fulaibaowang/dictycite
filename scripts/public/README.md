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
```

### plot_by_evidence_level.py

Generates pipeline-style plots stratified by **evidence_level** from the gold JSON: (1) hybrid recall curve with one subplot per evidence level; (2) rerank recall figure by evidence level; (3) rerank MAP@10 figure by evidence level. Saves to `workflow_dir/hybrid/figures/` and `workflow_dir/rerank/figures/` (or `--rerank-dir`).

**Usage:** Run from repo root with paths to the workflow output and gold JSON(s). See script docstring and `--help` for required arguments.


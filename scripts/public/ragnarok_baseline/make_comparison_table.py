#!/usr/bin/env python3
# Stack ragnarok-baseline metrics next to the project's pipeline metrics
# (BM25 / Dense / Fusion / Reranker) for the same goldset, in one table.
#
# Run AFTER both jobs finish:
#   - sbatch_pipeline_full_goldset.sh                    -> writes metrics.csv per stage
#   - sbatch_bm25_rerank.sh (this baseline)              -> writes metrics.csv with bm25 + rerank rows
#
# Usage:
#   python make_comparison_table.py \
#     --project_workflow_dir <WORKFLOW_OUTPUT_DIR>/retrieval \
#     --baseline_metrics_csv output/ragnarok_baseline/metrics.csv \
#     --output_csv output/ragnarok_baseline/comparison.csv

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_STAGES = {
    "bm25": "project_bm25",
    "dense": "project_dense",
    "fusion": "project_fusion",
    # rerank metrics are emitted by fuse_retrieval after the cross-encoder pass;
    # path varies by config so we look for it explicitly.
}


def read_project_metrics(workflow_retrieval_dir: Path):
    """Walk <workflow>/retrieval/{bm25,dense,fusion}/metrics.csv and tag the method."""
    rows = []
    for stage_dir, label in PROJECT_STAGES.items():
        mp = workflow_retrieval_dir / stage_dir / "metrics.csv"
        if not mp.exists():
            print(f"[compare] missing {mp}, skipping")
            continue
        df = pd.read_csv(mp)
        df["source"] = label
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def read_baseline_metrics(path: Path):
    df = pd.read_csv(path)
    df["source"] = df["method"].map(lambda m: f"ragnarok_{m}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_workflow_dir", required=True,
                    help="<WORKFLOW_OUTPUT_DIR>/retrieval directory from the project pipeline")
    ap.add_argument("--baseline_metrics_csv", required=True,
                    help="metrics.csv emitted by score_ragnarok_run.py")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--cols",
                    default="source,method,batch,n_queries,MRR@10,MAP@10,Success@10,"
                            "MeanR@10,MeanR@100,MeanR@500,MeanR@1000",
                    help="Comma-separated columns to keep in the comparison table")
    args = ap.parse_args()

    proj = read_project_metrics(Path(args.project_workflow_dir))
    base = read_baseline_metrics(Path(args.baseline_metrics_csv))

    if proj.empty and base.empty:
        raise SystemExit("No metrics found in either source.")

    combined = pd.concat([proj, base], ignore_index=True, sort=False)

    keep = [c for c in args.cols.split(",") if c in combined.columns]
    combined = combined[keep]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"[compare] wrote {out}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()

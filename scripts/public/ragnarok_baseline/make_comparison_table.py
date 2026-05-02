#!/usr/bin/env python3
# Stack ragnarok-baseline metrics next to all project pipeline stages
# (BM25 / Dense / Fusion / CrossEncoder / PostRerankFusion / Snippet variants)
# for the same goldset, in one normalised table.
#
# Usage:
#   python make_comparison_table.py \
#     --project_workflow_dir output/workflow_vega_7a_public_goldset_both_routes \
#     --baseline_metrics_csv output/ragnarok_baseline/metrics.csv \
#     --output_csv output/ragnarok_baseline/comparison.csv

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

N_QUERIES = 1656  # fallback when not stored in the csv

METRIC_COLS = [
    "MRR@10", "MAP@10", "GMAP@10", "Success@10",
    "MeanR@10", "MeanR@50", "MeanR@100", "MeanR@200",
    "MeanR@300", "MeanR@400", "MeanR@500", "MeanR@1000",
    "MeanR@2000", "MeanR@5000",
]


def _norm(df: pd.DataFrame, method: str, batch: str, n_queries: int) -> pd.DataFrame:
    """Return a single-row DataFrame with normalised columns."""
    row = {"method": method, "batch": batch, "n_queries": n_queries}
    for col in METRIC_COLS:
        row[col] = df[col].iloc[0] if col in df.columns else float("nan")
    return pd.DataFrame([row])


def read_standard(path: Path) -> pd.DataFrame | None:
    """retrieval/bm25 and retrieval/dense — already normalised."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[[0]]
    method = str(row["method"].iloc[0])
    batch = str(row["batch"].iloc[0])
    n = int(row["n_queries"].iloc[0]) if "n_queries" in row.columns else N_QUERIES
    return _norm(row, method, batch, n)


def read_fusion_best(path: Path) -> pd.DataFrame | None:
    """retrieval/fusion — pick the row with the highest MRR@10."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    best = df.loc[[df["MRR@10"].idxmax()]]
    batch = str(best["split"].iloc[0]) if "split" in best.columns else "unknown"
    return _norm(best, "Fusion (best)", batch, N_QUERIES)


def read_crossenc(path: Path) -> pd.DataFrame | None:
    """rerank/cross_encoder — schema: run, label, role, metrics..."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[[0]]
    batch = str(row["label"].iloc[0]) if "label" in row.columns else "unknown"
    return _norm(row, "CrossEncoder", batch, N_QUERIES)


def read_split_only(path: Path, method: str) -> pd.DataFrame | None:
    """post_rerank_fusion / snippet stages — schema: split, metrics..."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    # May have a label column (snippet_rerank does)
    row = df.iloc[[0]]
    batch = str(row.get("label", row.get("split", pd.Series(["unknown"]))).iloc[0])
    if "label" in row.columns:
        batch = str(row["label"].iloc[0])
    elif "split" in row.columns:
        batch = str(row["split"].iloc[0])
    return _norm(row, method, batch, N_QUERIES)


def read_baseline(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    rows = []
    for _, r in df.iterrows():
        label = f"Ragnarok/{r['method']}"
        batch = str(r.get("batch", ""))
        n = int(r.get("n_queries", N_QUERIES))
        rows.append(_norm(pd.DataFrame([r]), label, batch, n))
    return pd.concat(rows, ignore_index=True) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_workflow_dir", required=True,
                    help="top-level workflow output dir (contains retrieval/, rerank/, snippet/)")
    ap.add_argument("--baseline_metrics_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    wdir = Path(args.project_workflow_dir)

    parts = [
        read_standard(wdir / "retrieval/bm25/metrics.csv"),
        read_standard(wdir / "retrieval/dense/metrics.csv"),
        read_fusion_best(wdir / "retrieval/fusion/metrics.csv"),
        read_crossenc(wdir / "rerank/cross_encoder/metrics.csv"),
        read_split_only(wdir / "rerank/post_rerank_fusion/metrics.csv",        "PostRerankFusion"),
        read_split_only(wdir / "rerank/post_rerank_fusion_snippet/metrics.csv", "PostRerankFusion+Snippet"),
        read_split_only(wdir / "snippet/snippet_doc_fusion/metrics.csv",        "SnippetDocFusion"),
        read_split_only(wdir / "snippet/snippet_rerank/metrics.csv",            "SnippetRerank"),
        read_baseline(Path(args.baseline_metrics_csv)),
    ]

    frames = [p for p in parts if p is not None]
    if not frames:
        raise SystemExit("No metrics found.")

    combined = pd.concat(frames, ignore_index=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"[compare] wrote {out}\n")

    display_cols = ["method", "n_queries",
                    "MRR@10", "MAP@10", "Success@10",
                    "MeanR@10", "MeanR@100", "MeanR@500"]
    display = combined[[c for c in display_cols if c in combined.columns]]
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()

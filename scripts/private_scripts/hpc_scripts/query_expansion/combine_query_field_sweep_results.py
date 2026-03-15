#!/usr/bin/env python3
"""
Combine metrics from 3×3 query-field sweep runs (BM25 × Dense, optional Hybrid).

Reads bm25_*_dense_*/{bm25,dense}/metrics.csv and optionally hybrid/results_all.csv,
adds combo id and query-field labels, writes a single combined CSV with
train/test split preserved (batch column), and plots recall curves:
one figure per batch with curves for selected stages (bm25, dense, hybrid).

Usage:
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --workflow_dir output/workflow_hpc_test --no_plot
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --plot bm25 dense   # default
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --plot hybrid       # only hybrid
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --plot bm25 dense hybrid
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --log_x   # log-scale x-axis (K)
  python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --log_y   # log-scale y-axis (recall)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# Subdir pattern: bm25_<bm25_qf>_dense_<dense_qf>
COMBO_PATTERN = re.compile(r"^bm25_(body|synonyms|long)_dense_(body|synonyms|long)$")


def find_combo_dirs(workflow_dir: Path) -> list[tuple[str, str, Path]]:
    """Return list of (bm25_qf, dense_qf, path) for each bm25_*_dense_* subdir."""
    out = []
    if not workflow_dir.is_dir():
        return out
    for p in workflow_dir.iterdir():
        if not p.is_dir():
            continue
        m = COMBO_PATTERN.match(p.name)
        if m:
            out.append((m.group(1), m.group(2), p))
    return sorted(out, key=lambda x: (x[0], x[1]))


def load_stage_metrics(stage_path: Path, stage: str) -> pd.DataFrame | None:
    """Load metrics.csv from stage_path; return None if missing."""
    csv_path = stage_path / "metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df["stage"] = stage
    return df


def load_hybrid_metrics(hybrid_path: Path) -> pd.DataFrame | None:
    """Load hybrid results from results_all.csv; return None if missing.
    Normalizes to same schema as bm25/dense: batch (from split), stage, method, etc.
    """
    csv_path = hybrid_path / "results_all.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"split": "batch"})
    df["stage"] = "hybrid"
    df["method"] = "Hybrid"
    if "n_queries" not in df.columns:
        df["n_queries"] = pd.NA
    return df


def _recall_k_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted MeanR@* column names by K value."""
    cols = [c for c in df.columns if c.startswith("MeanR@")]
    cols.sort(key=lambda c: int(c.split("@")[1]))
    return cols


def _plot_recall_curves(
    combined: pd.DataFrame,
    output_dir: Path,
    *,
    plot_stages: tuple[str, ...] = ("bm25", "dense"),
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """One plot per batch: recall curves for selected stages (bm25, dense, hybrid)."""
    recall_cols = _recall_k_columns(combined)
    if not recall_cols:
        return
    ks = [int(c.split("@")[1]) for c in recall_cols]

    # Normalize and ensure string columns
    df = combined.copy()
    df["batch"] = df["batch"].astype(str).str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()
    df["combo"] = df["combo"].astype(str).str.strip()

    # Build one row per (batch, stage, qf) by grouping: use bm25_query_field for BM25, dense_query_field for Dense
    df["bm25_qf"] = df["bm25_query_field"].astype(str).str.strip()
    df["dense_qf"] = df["dense_query_field"].astype(str).str.strip()

    batches = df["batch"].unique().tolist()
    qf_order = ("body", "synonyms", "long")

    for batch in batches:
        fig, ax = plt.subplots()
        subset = df[df["batch"] == batch]
        if subset.empty:
            continue
        # BM25: one curve per bm25_qf (take first row for each value)
        if "bm25" in plot_stages:
            for qf in qf_order:
                rows = subset[(subset["stage"] == "bm25") & (subset["bm25_qf"] == qf)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                vals = [float(row[c]) for c in recall_cols]
                ax.plot(ks, vals, marker="s", label=f"BM25 ({qf})")
        # Dense: one curve per dense_qf
        if "dense" in plot_stages:
            for qf in qf_order:
                rows = subset[(subset["stage"] == "dense") & (subset["dense_qf"] == qf)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                vals = [float(row[c]) for c in recall_cols]
                ax.plot(ks, vals, marker="^", label=f"Dense ({qf})")
        # Hybrid: one curve per combo (bm25_qf, dense_qf)
        if "hybrid" in plot_stages:
            for bm25_qf in qf_order:
                for dense_qf in qf_order:
                    rows = subset[
                        (subset["stage"] == "hybrid")
                        & (subset["bm25_qf"] == bm25_qf)
                        & (subset["dense_qf"] == dense_qf)
                    ]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    vals = [float(row[c]) for c in recall_cols]
                    ax.plot(ks, vals, marker="o", label=f"Hybrid ({bm25_qf}, {dense_qf})")

        ax.set_xlabel("K")
        ax.set_ylabel("Mean Recall@K")
        ax.set_title(f"Recall curves ({batch})")
        ax.legend(fontsize="small")
        ax.set_ylim(0, 1.05)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        safe_name = re.sub(r"[^\w\-]", "_", str(batch))
        fig.savefig(output_dir / f"recall_curve_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine BM25 and Dense metrics from 3×3 query-field sweep (train/test split preserved)."
    )
    ap.add_argument(
        "--workflow_dir",
        type=Path,
        default=Path("output/workflow_hpc_test"),
        help="Base workflow directory containing bm25_*_dense_* subdirs",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <workflow_dir>/combined_sweep_metrics.csv)",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip recall curve plots",
    )
    ap.add_argument(
        "--figures_dir",
        type=Path,
        default=None,
        help="Directory for recall curve PNGs (default: <workflow_dir>/figures)",
    )
    ap.add_argument(
        "--log_x",
        action="store_true",
        help="Use log scale for x-axis (K)",
    )
    ap.add_argument(
        "--log_y",
        action="store_true",
        help="Use log scale for y-axis (Mean Recall@K)",
    )
    ap.add_argument(
        "--plot",
        nargs="+",
        choices=["bm25", "dense", "hybrid"],
        default=["bm25", "dense"],
        metavar="STAGE",
        help="Stages to load and plot: bm25, dense, hybrid (default: bm25 dense)",
    )
    args = ap.parse_args()

    workflow_dir = args.workflow_dir.resolve()
    out_path = args.out or (workflow_dir / "combined_sweep_metrics.csv")

    combos = find_combo_dirs(workflow_dir)
    if not combos:
        print(f"No bm25_*_dense_* combo dirs found under {workflow_dir}")
        return

    plot_stages = tuple(args.plot)
    rows = []
    for bm25_qf, dense_qf, combo_path in combos:
        combo_name = combo_path.name
        for stage in ("bm25", "dense"):
            if stage not in plot_stages:
                continue
            stage_path = combo_path / stage
            df = load_stage_metrics(stage_path, stage)
            if df is None:
                print(f"Skip (missing): {combo_name}/{stage}/metrics.csv")
                continue
            df["combo"] = combo_name
            df["bm25_query_field"] = bm25_qf
            df["dense_query_field"] = dense_qf
            rows.append(df)
        if "hybrid" in plot_stages:
            hybrid_path = combo_path / "hybrid"
            df = load_hybrid_metrics(hybrid_path)
            if df is None:
                print(f"Skip (missing): {combo_name}/hybrid/results_all.csv")
            else:
                df["combo"] = combo_name
                df["bm25_query_field"] = bm25_qf
                df["dense_query_field"] = dense_qf
                rows.append(df)

    if not rows:
        print("No metrics loaded.")
        return

    combined = pd.concat(rows, ignore_index=True)

    # Column order: combo and query fields first, then batch/split, then stage/method, then metrics
    id_cols = ["combo", "bm25_query_field", "dense_query_field", "stage", "method", "batch", "n_queries"]
    rest = [c for c in combined.columns if c not in id_cols]
    combined = combined[[c for c in id_cols if c in combined.columns] + rest]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}")
    print(f"Combos: {len(combos)}  Stages: {list(plot_stages)}  Batches: {combined['batch'].unique().tolist()}")

    if not args.no_plot and _HAS_MATPLOTLIB:
        figures_dir = args.figures_dir or (workflow_dir / "figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        _plot_recall_curves(
            combined,
            figures_dir,
            plot_stages=plot_stages,
            log_x=args.log_x,
            log_y=args.log_y,
        )
        print(f"Recall curves saved to {figures_dir}")
    elif args.no_plot:
        print("Skipping plots (--no_plot)")
    else:
        print("Skipping plots (matplotlib not available)")


if __name__ == "__main__":
    main()

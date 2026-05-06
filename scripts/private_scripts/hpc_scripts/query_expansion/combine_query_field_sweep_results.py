#!/usr/bin/env python3
"""
Combine metrics from 3×3 query-field sweep or from rerank sweep runs.

Two layouts supported:
  1) 3×3 sweep: workflow_dir contains bm25_*_dense_* subdirs; each has bm25/, dense/, hybrid/.
  2) Rerank sweep (--rerank_sweep): workflow_dir contains bm25/, dense/, hybrid/, rerank_body/, rerank_synonyms/, rerank_long/.

Reads metrics.csv (and hybrid results_all.csv), adds combo/stage labels, writes one combined CSV,
and optionally plots recall curves (one figure per batch).

Note: In the 3×3 sweep, dense metrics depend only on dense_query_field (body/synonyms/long), not on
bm25_query_field, because each retriever uses its own query field. So in the CSV, all rows with
the same (stage=dense, dense_query_field=X) have identical metric values across bm25_query_field.
This is expected; the figures still show the correct per-combo curves.

Usage:
  # 3×3 query-field sweep (default)
  python .../combine_query_field_sweep_results.py --workflow_dir output/workflow_baseline_full_sweep/query_field_sweep
  # Rerank sweep (fixed long + rerank body/synonyms/long)
  python .../combine_query_field_sweep_results.py --rerank_sweep --workflow_dir output/workflow_baseline_full_sweep/fixed_long_rerank_sweep
  # Optional: same on Frida if you want combined_sweep_metrics.csv / combine's recall PNGs (not required for rerank-only MRR plots in query_expansion_sweeping.ipynb).
  python .../combine_query_field_sweep_results.py --rerank_sweep --workflow_dir /shared/workspace/biolab/yun/dicty_output/workflow_fixed_long_rerank_sweep_7d_medcpt/fixed_long_rerank_sweep
  python .../combine_query_field_sweep_results.py --workflow_dir output/workflow_hpc_test --no_plot
  python .../combine_query_field_sweep_results.py --plot bm25 dense hybrid
  python .../combine_query_field_sweep_results.py --log_x --log_y
  python .../combine_query_field_sweep_results.py --wide   # also write 3-row wide CSV (one per query_field)
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

# Rerank sweep layout: workflow_dir contains bm25/, dense/, hybrid/, rerank_body/, rerank_synonyms/, rerank_long/
RERANK_SWEEP_STAGES = ("bm25", "dense", "hybrid", "rerank_body", "rerank_synonyms", "rerank_long")


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
    """Load hybrid results from results_all.csv (legacy) or metrics.csv (new fusion layout).

    Normalizes to same schema as bm25/dense: batch (from split), stage, method, etc.
    """
    csv_path = hybrid_path / "results_all.csv"
    if not csv_path.is_file():
        csv_path = hybrid_path / "metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if "split" in df.columns and "batch" not in df.columns:
        df = df.rename(columns={"split": "batch"})
    df["stage"] = "hybrid"
    df["method"] = "Hybrid"
    if "n_queries" not in df.columns:
        df["n_queries"] = pd.NA
    return df


def resolve_stage_path(parent: Path, stage: str) -> Path:
    """Return the directory for a stage, supporting both legacy and new layouts.

    Legacy: <parent>/<stage>/  (hybrid lives at <parent>/hybrid/)
    New:    <parent>/retrieval/<stage>/  (hybrid lives at <parent>/retrieval/fusion/)
    """
    legacy_subdir = stage
    new_subdir = "fusion" if stage == "hybrid" else stage
    new_path = parent / "retrieval" / new_subdir
    if new_path.is_dir():
        return new_path
    return parent / legacy_subdir


def find_rerank_sweep_stages(workflow_dir: Path) -> list[str]:
    """Return list of stage names that exist under workflow_dir (fixed_long_rerank_sweep layout)."""
    found = []
    for stage in RERANK_SWEEP_STAGES:
        stage_path = resolve_stage_path(workflow_dir, stage)
        if stage == "hybrid":
            if (stage_path / "results_all.csv").is_file() or (stage_path / "metrics.csv").is_file():
                found.append(stage)
        else:
            if (stage_path / "metrics.csv").is_file():
                found.append(stage)
    return found


def load_rerank_sweep_rows(workflow_dir: Path, stages: list[str]) -> list[pd.DataFrame]:
    """Load metrics from each stage subdir; return list of DataFrames with combo and query-field columns set."""
    rows = []
    for stage in stages:
        stage_path = resolve_stage_path(workflow_dir, stage)
        if stage == "hybrid":
            df = load_hybrid_metrics(stage_path)
        else:
            df = load_stage_metrics(stage_path, stage)
        if df is None:
            continue
        df["combo"] = "fixed_long"
        df["bm25_query_field"] = "long"
        df["dense_query_field"] = "long"
        rows.append(df)
    return rows


def _batch_short(batch: str) -> str:
    """Return 'train' or 'test' from batch name."""
    b = str(batch).strip().lower()
    return "train" if "train" in b else "test"


def _long_to_wide_3x3(combined: pd.DataFrame) -> pd.DataFrame:
    """Collapse 3×3 sweep to 3 rows (one per query_field): BM25 and Dense metrics side by side.

    BM25 metrics depend only on bm25_query_field, Dense only on dense_query_field,
    so the non-redundant representation is 3 rows keyed by query_field, with columns:
      query_field, {train|test}_bm25_{metric}, {train|test}_dense_{metric}
    """
    id_cols = ["combo", "bm25_query_field", "dense_query_field"]
    meta = ["stage", "method", "batch", "n_queries"]
    metric_cols = [c for c in combined.columns if c not in id_cols and c not in meta]
    if not metric_cols:
        qfs = sorted(set(combined["bm25_query_field"].unique()) | set(combined["dense_query_field"].unique()))
        return pd.DataFrame({"query_field": qfs})

    combined = combined.copy()
    combined["_batch_short"] = combined["batch"].map(_batch_short)

    query_fields = sorted(set(combined["bm25_query_field"].unique()) | set(combined["dense_query_field"].unique()))
    wide_rows = []
    for qf in query_fields:
        row: dict[str, object] = {"query_field": qf}
        bm25_rows = combined[(combined["stage"] == "bm25") & (combined["bm25_query_field"] == qf)]
        for _, r in bm25_rows.drop_duplicates(subset=["_batch_short"]).iterrows():
            prefix = f"{r['_batch_short']}_bm25_"
            for col in metric_cols:
                row[prefix + col] = r[col]
        dense_rows = combined[(combined["stage"] == "dense") & (combined["dense_query_field"] == qf)]
        for _, r in dense_rows.drop_duplicates(subset=["_batch_short"]).iterrows():
            prefix = f"{r['_batch_short']}_dense_"
            for col in metric_cols:
                row[prefix + col] = r[col]
        wide_rows.append(row)
    out = pd.DataFrame(wide_rows)
    rest = [c for c in out.columns if c != "query_field"]
    rest.sort()
    out = out[["query_field"] + rest]
    return out


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


def _plot_recall_curves_rerank_sweep(
    combined: pd.DataFrame,
    output_dir: Path,
    *,
    plot_stages: tuple[str, ...],
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """One plot per batch: one curve per stage (for fixed_long_rerank_sweep layout)."""
    recall_cols = _recall_k_columns(combined)
    if not recall_cols:
        return
    ks = [int(c.split("@")[1]) for c in recall_cols]
    df = combined.copy()
    df["batch"] = df["batch"].astype(str).str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()
    batches = df["batch"].unique().tolist()
    for batch in batches:
        fig, ax = plt.subplots()
        subset = df[df["batch"] == batch]
        if subset.empty:
            continue
        for stage in plot_stages:
            rows = subset[subset["stage"] == stage]
            if rows.empty:
                continue
            row = rows.iloc[0]
            vals = [float(row[c]) for c in recall_cols]
            ax.plot(ks, vals, marker="o", label=stage)
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
        help="Base workflow directory (bm25_*_dense_* subdirs, or with --rerank_sweep: bm25/, dense/, hybrid/, rerank_*)",
    )
    ap.add_argument(
        "--rerank_sweep",
        action="store_true",
        help="Treat workflow_dir as fixed_long_rerank_sweep (stage subdirs bm25, dense, hybrid, rerank_body, rerank_synonyms, rerank_long).",
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
        "--wide",
        action="store_true",
        help="Also write a compact 3-row CSV (one per query_field) with BM25 and Dense metrics side by side (3×3 sweep only).",
    )
    all_stages = list(RERANK_SWEEP_STAGES)
    ap.add_argument(
        "--plot",
        nargs="+",
        choices=all_stages,
        default=None,
        metavar="STAGE",
        help="Stages to load and plot (default: bm25 dense for 3×3; all found stages for --rerank_sweep).",
    )
    args = ap.parse_args()

    workflow_dir = args.workflow_dir.resolve()
    out_path = args.out or (workflow_dir / "combined_sweep_metrics.csv")

    if args.rerank_sweep:
        stages = find_rerank_sweep_stages(workflow_dir)
        if not stages:
            print(f"No rerank-sweep stage dirs found under {workflow_dir} (expected bm25/, dense/, hybrid/, rerank_*)")
            return
        plot_stages = tuple(args.plot) if args.plot else stages
        rows = load_rerank_sweep_rows(workflow_dir, stages)
        if not rows:
            print("No metrics loaded from rerank sweep dir.")
            return
        combined = pd.concat(rows, ignore_index=True)
        id_cols = ["combo", "bm25_query_field", "dense_query_field", "stage", "method", "batch", "n_queries"]
        rest = [c for c in combined.columns if c not in id_cols]
        combined = combined[[c for c in id_cols if c in combined.columns] + rest]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"Wrote {len(combined)} rows to {out_path}")
        print(f"Rerank sweep stages: {stages}  Batches: {combined['batch'].unique().tolist()}")
        if not args.no_plot and _HAS_MATPLOTLIB:
            figures_dir = args.figures_dir or (workflow_dir / "figures")
            figures_dir.mkdir(parents=True, exist_ok=True)
            _plot_recall_curves_rerank_sweep(
                combined,
                figures_dir,
                plot_stages=tuple(s for s in plot_stages if s in combined["stage"].unique()),
                log_x=args.log_x,
                log_y=args.log_y,
            )
            print(f"Recall curves saved to {figures_dir}")
        elif args.no_plot:
            print("Skipping plots (--no_plot)")
        else:
            print("Skipping plots (matplotlib not available)")
        return

    # 3×3 query-field sweep layout
    combos = find_combo_dirs(workflow_dir)
    if not combos:
        print(f"No bm25_*_dense_* combo dirs found under {workflow_dir}")
        return

    plot_stages = tuple(args.plot) if args.plot else ("bm25", "dense")
    rows = []
    for bm25_qf, dense_qf, combo_path in combos:
        combo_name = combo_path.name
        for stage in ("bm25", "dense"):
            if stage not in plot_stages:
                continue
            stage_path = resolve_stage_path(combo_path, stage)
            df = load_stage_metrics(stage_path, stage)
            if df is None:
                print(f"Skip (missing): {combo_name}/{stage}/metrics.csv")
                continue
            df["combo"] = combo_name
            df["bm25_query_field"] = bm25_qf
            df["dense_query_field"] = dense_qf
            rows.append(df)
        if "hybrid" in plot_stages:
            hybrid_path = resolve_stage_path(combo_path, "hybrid")
            df = load_hybrid_metrics(hybrid_path)
            if df is None:
                print(f"Skip (missing): {combo_name}/(retrieval/fusion|hybrid)/(metrics|results_all).csv")
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

    if args.wide:
        wide_df = _long_to_wide_3x3(combined)
        wide_path = out_path.parent / (out_path.stem + "_wide" + out_path.suffix)
        wide_df.to_csv(wide_path, index=False)
        print(f"Wrote {len(wide_df)} rows (wide, one per query_field) to {wide_path}")

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

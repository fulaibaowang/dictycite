#!/usr/bin/env python3
"""
Generate pipeline-style plots stratified by evidence_level (from gold JSON).
Produces: (1) hybrid recall curve with one subplot per evidence_level;
(2) rerank recall figure with one subplot per evidence_level;
(3) rerank MAP@10 figure with one subplot per evidence_level.
Saves to workflow_dir/hybrid/figures/ and workflow_dir/rerank/figures/ (or --rerank-dir).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]  # file -> public -> scripts -> repo
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "public" / "shared_scripts"))

from retrieval_eval.common import (
    build_topics_and_gold,
    evaluate_run,
    load_questions,
    normalize_pmid,
    run_df_to_run_map,
)
from rerank.plot_rerank_eval import _load_run_tsv, _meanr_columns_to_k_list


def build_qid_to_evidence_level(gold_json: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Load gold JSON and return (gold_map, qid_to_evidence_level).
    evidence_level is taken from the first doc in `docs` per question (order matches documents).
    Questions with no docs get evidence_level "unknown".
    """
    data = json.loads(gold_json.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    gold_map: Dict[str, List[str]] = {}
    qid_to_level: Dict[str, str] = {}
    for i, q in enumerate(questions):
        qid = str(q.get("id") or q.get("qid") or i)
        docs = q.get("docs") or []
        documents = q.get("documents") or []
        pmids = [normalize_pmid(d) for d in documents]
        pmids = [p for p in pmids if p]
        gold_map[qid] = pmids
        if docs and len(docs) > 0:
            level = (docs[0].get("evidence_level") or "unknown").strip() or "unknown"
        else:
            level = "unknown"
        qid_to_level[qid] = level
    return gold_map, qid_to_level


def subset_by_evidence_level(
    gold_map: Dict[str, List[str]],
    qid_to_level: Dict[str, str],
    run_map: Dict[str, List[str]],
    levels_order: Optional[List[str]] = None,
) -> Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]]]]:
    """
    For each evidence_level, return (gold_subset, run_subset) for qids that have that level.
    Skips "unknown" unless present in levels_order. Returns dict keyed by level.
    """
    level_to_qids: Dict[str, List[str]] = {}
    for qid, level in qid_to_level.items():
        if qid not in gold_map or qid not in run_map:
            continue
        level_to_qids.setdefault(level, []).append(qid)
    if levels_order is None:
        levels_order = sorted(level_to_qids.keys())
    out: Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]]]] = {}
    for level in levels_order:
        qids = level_to_qids.get(level, [])
        if not qids:
            continue
        gold_sub = {q: gold_map[q] for q in qids}
        run_sub = {q: run_map[q] for q in qids}
        out[level] = (gold_sub, run_sub)
    return out


def _split_name_from_run_stem(stem: str) -> str:
    """e.g. best_rrf_train_subset_top5000 -> train_subset; best_rrf_foo_top5000 -> foo."""
    if stem.startswith("best_rrf_") and "_top" in stem:
        return stem.replace("best_rrf_", "").split("_top")[0]
    return stem


def plot_hybrid_by_evidence_level(
    workflow_dir: Path,
    gold_map: Dict[str, List[str]],
    qid_to_level: Dict[str, str],
    split: str,
    ks: List[int],
    best_cfg: Dict,
    bm25_run: Dict[str, List[str]],
    dense_run: Dict[str, List[str]],
    hybrid_run: Dict[str, List[str]],
    k_max_eval_eff: int,
    p: float,
    output_suffix: Optional[str] = None,
) -> None:
    """One figure: subplots per evidence_level (recall curve style), shared y-axis. output_suffix used in filename."""
    if plt is None:
        return
    levels_order = sorted(set(qid_to_level.values()) - {"unknown"})
    if not levels_order:
        levels_order = sorted(set(qid_to_level.values()))
    by_level = subset_by_evidence_level(gold_map, qid_to_level, hybrid_run, levels_order=levels_order)
    if not by_level:
        print("No evidence_level subsets with overlapping qids; skipping hybrid figure.")
        return
    n = len(by_level)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    k_rrf = int(best_cfg.get("k_rrf", 60))
    y_min, y_max = 1.0, 0.0
    for ax, (level, (gold_sub, run_sub)) in zip(axes, by_level.items()):
        hybrid_metrics, _ = evaluate_run(gold_sub, run_sub, ks_recall=tuple(ks))
        bm25_sub = {q: bm25_run[q] for q in gold_sub if q in bm25_run}
        dense_sub = {q: dense_run[q] for q in gold_sub if q in dense_run}
        bm25_metrics, _ = evaluate_run(gold_sub, bm25_sub, ks_recall=tuple(ks)) if bm25_sub else ({}, pd.DataFrame())
        dense_metrics, _ = evaluate_run(gold_sub, dense_sub, ks_recall=tuple(ks)) if dense_sub else ({}, pd.DataFrame())
        ax.plot(
            ks, [hybrid_metrics.get(f"MeanR@{k}", np.nan) for k in ks],
            marker="o", label=f"Hybrid (k_rrf={k_rrf})", color="#444444",
        )
        ax.plot(ks, [bm25_metrics.get(f"MeanR@{k}", np.nan) for k in ks], marker="s", label="BM25")
        ax.plot(ks, [dense_metrics.get(f"MeanR@{k}", np.nan) for k in ks], marker="^", label="Dense")
        rmax = hybrid_metrics.get(f"MeanR@{ks[-1]}", np.nan) if ks else np.nan
        if np.isfinite(rmax):
            ax.axhline(p * rmax, linestyle="--", label=f"p*Rmax (p={p})")
        ax.set_xlabel("K")
        ax.set_ylabel("Mean Recall@K")
        ax.set_title(f"{level} (n={len(gold_sub)})")
        ax.legend(fontsize=8)
        for d in (hybrid_metrics, bm25_metrics, dense_metrics):
            for k in ks:
                v = d.get(f"MeanR@{k}", np.nan)
                if np.isfinite(v):
                    y_min = min(y_min, v)
                    y_max = max(y_max, v)
    for ax in axes:
        ax.set_ylim(max(0, y_min - 0.05), min(1.0, y_max + 0.05))
    plt.tight_layout()
    out_dir = workflow_dir / "hybrid" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (output_suffix or split).replace("/", "_").replace("\\", "_") or "all"
    path = out_dir / f"recall_curve_by_evidence_level_{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def plot_rerank_by_evidence_level(
    rerank_dir: Path,
    gold_map: Dict[str, List[str]],
    qid_to_level: Dict[str, str],
    hybrid_run: Dict[str, List[str]],
    reranker_run: Dict[str, List[str]],
    run_id: str,
    k_list: List[int],
    output_suffix: Optional[str] = None,
) -> None:
    """Two figures: (1) Recall subplots per evidence_level, (2) MAP@10 subplots per evidence_level. Shared y-axis each."""
    if plt is None:
        return
    levels_order = sorted(set(qid_to_level.values()) - {"unknown"})
    if not levels_order:
        levels_order = sorted(set(qid_to_level.values()))
    by_level = subset_by_evidence_level(gold_map, qid_to_level, hybrid_run, levels_order=levels_order)
    if not by_level:
        print("No evidence_level subsets for rerank; skipping.")
        return
    n = len(by_level)
    colors = {"Hybrid": "#444444", "Reranker": "#1f77b4"}
    # Figure 1: Recall (per-level hybrid vs reranker from actual runs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    y_min_r, y_max_r = 1.0, 0.0
    for ax, (level, (gold_sub, hybrid_sub)) in zip(axes, by_level.items()):
        reranker_sub = {q: reranker_run[q] for q in gold_sub if q in reranker_run}
        hybrid_metrics, _ = evaluate_run(gold_sub, hybrid_sub, ks_recall=tuple(k_list))
        reranker_metrics, _ = evaluate_run(gold_sub, reranker_sub, ks_recall=tuple(k_list))
        ax.plot(k_list, [hybrid_metrics.get(f"MeanR@{k}", np.nan) for k in k_list], marker="o", label="Hybrid", color=colors["Hybrid"])
        ax.plot(k_list, [reranker_metrics.get(f"MeanR@{k}", np.nan) for k in k_list], marker="o", label="Reranker", color=colors["Reranker"])
        ax.set_xlabel("K (Recall Cutoff)")
        ax.set_ylabel("Mean Recall")
        ax.set_title(f"{level} (n={len(gold_sub)})")
        ax.legend(fontsize=9)
        for d in (hybrid_metrics, reranker_metrics):
            for v in d.values():
                if np.isfinite(v):
                    y_min_r = min(y_min_r, v)
                    y_max_r = max(y_max_r, v)
    for ax in axes:
        ax.set_ylim(max(0, y_min_r - 0.05), min(1.0, y_max_r + 0.05))
    plt.tight_layout()
    figures_dir = rerank_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = (output_suffix or run_id).replace("/", "_").replace("\\", "_") or "all"
    plt.savefig(figures_dir / f"hybrid_reranker_recall_by_evidence_level_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", figures_dir / f"hybrid_reranker_recall_by_evidence_level_{suffix}.png")
    # Figure 2: MAP@10
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes2 = [axes2]
    y_min_m, y_max_m = 1.0, 0.0
    for ax, (level, (gold_sub, hybrid_sub)) in zip(axes2, by_level.items()):
        reranker_sub = {q: reranker_run[q] for q in gold_sub if q in reranker_run}
        hybrid_metrics, _ = evaluate_run(gold_sub, hybrid_sub, ks_recall=tuple(k_list))
        reranker_metrics, _ = evaluate_run(gold_sub, reranker_sub, ks_recall=tuple(k_list))
        map_hybrid = hybrid_metrics.get("MAP@10", np.nan)
        map_rerank = reranker_metrics.get("MAP@10", np.nan)
        ax.bar(["Hybrid", "Reranker"], [map_hybrid, map_rerank], color=[colors["Hybrid"], colors["Reranker"]])
        ax.set_ylabel("MAP@10")
        ax.set_title(f"{level} (n={len(gold_sub)})")
        ax.tick_params(axis="x", rotation=25)
        for v in [map_hybrid, map_rerank]:
            if np.isfinite(v):
                y_min_m = min(y_min_m, v)
                y_max_m = max(y_max_m, v)
    for ax in axes2:
        ax.set_ylim(max(0, y_min_m - 0.05), min(1.0, y_max_m + 0.05))
    plt.tight_layout()
    plt.savefig(figures_dir / f"hybrid_reranker_map10_by_evidence_level_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", figures_dir / f"hybrid_reranker_map10_by_evidence_level_{suffix}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pipeline metrics stratified by evidence_level.")
    ap.add_argument("--workflow-dir", type=Path, default=None, help="Workflow output dir (default: WORKFLOW_OUTPUT_DIR).")
    ap.add_argument("--gold-json", type=Path, required=True, help="Gold questions JSON with docs[].evidence_level.")
    ap.add_argument("--rerank-dir", type=str, default="rerank", help="Rerank subdir under workflow-dir (e.g. rerank_bge).")
    ap.add_argument("--test-batch-json", type=Path, default=None, help="Optional: filter to qids in this batch and use its run.")
    args = ap.parse_args()

    import os
    if args.workflow_dir is not None:
        workflow_dir = Path(args.workflow_dir).resolve()
    elif os.environ.get("WORKFLOW_OUTPUT_DIR"):
        workflow_dir = Path(os.environ["WORKFLOW_OUTPUT_DIR"]).resolve()
    else:
        workflow_dir = Path.cwd().resolve()
    gold_json = args.gold_json.resolve()
    if not gold_json.exists():
        raise FileNotFoundError(f"Gold JSON not found: {gold_json}")

    gold_map_full, qid_to_level_full = build_qid_to_evidence_level(gold_json)

    hybrid_dir = workflow_dir / "hybrid"
    rerank_dir = workflow_dir / args.rerank_dir
    if not hybrid_dir.exists():
        raise FileNotFoundError(f"Hybrid dir not found: {hybrid_dir}")

    runs_dir = hybrid_dir / "runs"
    run_files = sorted(runs_dir.glob("*.tsv")) if runs_dir.exists() else []
    if not run_files:
        raise FileNotFoundError(f"No TSV runs in {runs_dir}")
    run_maps = {path.stem: run_df_to_run_map(_load_run_tsv(path)) for path in run_files}

    if args.test_batch_json and args.test_batch_json.exists():
        filter_qids = set()
        for i, q in enumerate(load_questions(args.test_batch_json)):
            filter_qids.add(str(q.get("id") or q.get("qid") or i))
        run_stems_to_process = [s for s, r in run_maps.items() if filter_qids.intersection(r) and filter_qids.intersection(gold_map_full)]
        if not run_stems_to_process:
            run_stems_to_process = list(run_maps.keys())
    else:
        run_stems_to_process = list(run_maps.keys())

    best_cfg_path = hybrid_dir / "best_config.json"
    config_path = hybrid_dir / "config.json"
    if not best_cfg_path.exists():
        raise FileNotFoundError(f"Best config not found: {best_cfg_path}")
    best_cfg = json.loads(best_cfg_path.read_text(encoding="utf-8"))
    ks = list(best_cfg.get("ks_eval", [50, 100, 200, 300, 400, 500, 1000, 2000, 5000]))
    if not ks and config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        ks = cfg.get("ks_eval", [50, 100, 200, 300, 400, 500, 1000, 2000, 5000])
    if not ks:
        ks = [50, 100, 200, 300, 400, 500, 1000, 2000, 5000]
    k_max_eval_eff = max(ks) if ks else 5000
    p = float(best_cfg.get("p", 0.95))

    bm25_dir = workflow_dir / "bm25" / "runs"
    dense_dir = workflow_dir / "dense"
    bm25_method = "BM25_RM3"
    bm25_topk = 5000
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        bm25_method = cfg.get("bm25_method", "BM25_RM3")
        bm25_topk = int(cfg.get("bm25_topk", 5000))

    for run_stem in run_stems_to_process:
        hybrid_run = run_maps[run_stem]
        split_name = _split_name_from_run_stem(run_stem)
        qids_in_run = set(hybrid_run.keys())
        gold_map = {q: gold_map_full[q] for q in qids_in_run if q in gold_map_full}
        qid_to_level = {q: qid_to_level_full[q] for q in gold_map if q in qid_to_level_full}
        if not gold_map:
            continue
        bm25_path = bm25_dir / f"{bm25_method}__{split_name}__top{bm25_topk}.tsv"
        dense_tsv = dense_dir / "runs" / f"dense_{split_name}.tsv"
        dense_pq = dense_dir / f"dense_{split_name}.parquet"
        bm25_run: Dict[str, List[str]] = {}
        dense_run: Dict[str, List[str]] = {}
        if bm25_path.exists():
            bm25_run = run_df_to_run_map(_load_run_tsv(bm25_path))
        if dense_tsv.exists():
            dense_df = _load_run_tsv(dense_tsv)
            dense_run = run_df_to_run_map(dense_df[dense_df["rank"] <= k_max_eval_eff])
        elif dense_pq.exists():
            dense_df = pd.read_parquet(dense_pq)
            dense_df["qid"] = dense_df["qid"].astype(str)
            dense_df["docno"] = dense_df["docno"].astype(str).map(normalize_pmid)
            dense_run = run_df_to_run_map(dense_df[dense_df["rank"] <= k_max_eval_eff])
        plot_hybrid_by_evidence_level(
            workflow_dir=workflow_dir,
            gold_map=gold_map,
            qid_to_level=qid_to_level,
            split=split_name,
            ks=ks,
            best_cfg=best_cfg,
            bm25_run=bm25_run,
            dense_run=dense_run,
            hybrid_run=hybrid_run,
            k_max_eval_eff=k_max_eval_eff,
            p=p,
            output_suffix=split_name,
        )

    if rerank_dir.exists():
        metrics_path = rerank_dir / "metrics.csv"
        rerank_runs_dir = rerank_dir / "runs"
        if metrics_path.exists() and rerank_runs_dir.exists():
            metrics_df = pd.read_csv(metrics_path)
            k_list = _meanr_columns_to_k_list(metrics_df)
            candidate_limit = None
            cfg_path = rerank_dir / "config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                candidate_limit = cfg.get("candidate_limit")
            if candidate_limit and k_list:
                k_list = [k for k in k_list if k <= candidate_limit]
            if k_list:
                reranker_run_maps = {p.stem: run_df_to_run_map(_load_run_tsv(p)) for p in rerank_runs_dir.glob("*.tsv")}
                for run_id in run_stems_to_process:
                    if run_id not in run_maps or run_id not in reranker_run_maps:
                        continue
                    qids_in_run = set(run_maps[run_id].keys())
                    gold_map = {q: gold_map_full[q] for q in qids_in_run if q in gold_map_full}
                    qid_to_level = {q: qid_to_level_full[q] for q in gold_map if q in qid_to_level_full}
                    if not gold_map:
                        continue
                    split_name = _split_name_from_run_stem(run_id)
                    plot_rerank_by_evidence_level(
                        rerank_dir=rerank_dir,
                        gold_map=gold_map,
                        qid_to_level=qid_to_level,
                        hybrid_run=run_maps[run_id],
                        reranker_run=reranker_run_maps[run_id],
                        run_id=run_id,
                        k_list=k_list,
                        output_suffix=split_name,
                    )
    print("Done.")


if __name__ == "__main__":
    main()

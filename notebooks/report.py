# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dicty (Python 3.14 venv)
#     language: python
#     name: dicty-py314
# ---

# %% [markdown]
# # Workflow Baseline Full Run – Both Routes (Query Expansion)
# Plots built from `output/workflow_baseline_full_run_both_routes_query_expansion/{bm25,dense,hybrid,...}`.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# %% [markdown]
# ## 2. Load Data

# %%
base_dir = Path.cwd().resolve()
if not (base_dir / "output").exists() and (base_dir.parent / "output").exists():
    base_dir = base_dir.parent

print("Base dir:", base_dir)

workflow_dir = base_dir / "output" / "workflow_baseline_full_run_both_routes_query_expansion"

bm25_metrics = pd.read_csv(workflow_dir / "bm25" / "metrics.csv")
dense_metrics = pd.read_csv(workflow_dir / "dense" / "metrics.csv")
hybrid_metrics = pd.read_csv(workflow_dir / "hybrid" / "metrics.csv")

bm25_metrics = bm25_metrics.rename(columns={"batch": "split"})
dense_metrics = dense_metrics.rename(columns={"batch": "split"})

bm25_metrics["method"] = "BM25"
dense_metrics["method"] = "Dense"
hybrid_metrics["method"] = "BM25 Dense Fusion"

output_dir = workflow_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

splits = [
    "dicty_gold_llm_public_train_200",
    "dicty_gold_llm_public_test_50",
]
split_labels = {
    "dicty_gold_llm_public_train_200": "dev",
    "dicty_gold_llm_public_test_50": "test",
}

print("Loaded BM25:", bm25_metrics.shape)
print("Loaded Dense:", dense_metrics.shape)
print("Loaded Hybrid:", hybrid_metrics.shape)

# %% [markdown]
# ## 3. Stage 1 Recall Curves – Per Split (1×2 panels)

# %%
recall_cols_bm25 = sorted(
    [c for c in bm25_metrics.columns if c.startswith("MeanR@")],
    key=lambda c: int(c.split("@")[1]),
)
recall_cols_hybrid = sorted(
    [c for c in hybrid_metrics.columns if c.startswith("MeanR@")],
    key=lambda c: int(c.split("@")[1]),
)
recall_cols = sorted(
    set(recall_cols_bm25) & set(recall_cols_hybrid),
    key=lambda c: int(c.split("@")[1]),
)
recall_cols = [c for c in recall_cols if int(c.split("@")[1]) <= 2000]
k_values = [int(c.split("@")[1]) for c in recall_cols]

print("Shared K values (plotted up to 2000):", k_values)

tick_candidates = [50, 200, 500, 1000, 2000]
tick_values = [k for k in tick_candidates if k in k_values]

methods_cfg = {
    "BM25": {"df": bm25_metrics, "color": "#1f77b4", "marker": "o"},
    "Dense": {"df": dense_metrics, "color": "#ff7f0e", "marker": "s"},
    "BM25 Dense Fusion": {"df": hybrid_metrics, "color": "#2ca02c", "marker": "D"},
}

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)

global_ymin, global_ymax = 1.0, 0.0
for split in splits:
    for cfg in methods_cfg.values():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols]
        global_ymin = min(global_ymin, min(vals))
        global_ymax = max(global_ymax, max(vals))

y_pad = (global_ymax - global_ymin) * 0.05
global_ymin = max(0, global_ymin - y_pad)
#global_ymax = min(1, global_ymax + y_pad)
global_ymax = 1.02

for idx, split in enumerate(splits):
    ax = axes[idx]
    for method_name, cfg in methods_cfg.items():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols]
        ax.plot(
            k_values,
            vals,
            marker=cfg["marker"],
            color=cfg["color"],
            label=method_name,
            markersize=6,
            linewidth=1.8,
        )

    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(global_ymin, global_ymax)

    if idx == 0:
        ax.set_ylabel("Mean Recall")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("K")
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(k) for k in tick_values], rotation=90)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.15), fontsize=14)

fig.suptitle("Retrieval Mean_Recall@K Curves", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "01_stage1_recall_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 4. Retrieval vs Rerank – Recall Curves (K ≤ 300)

# %%
hybrid_stage1 = hybrid_metrics.copy()
rerank_metrics = pd.read_csv(workflow_dir / "rerank" / "metrics.csv")
rerank_fusion_metrics = pd.read_csv(workflow_dir / "rerank_hybrid_200" / "metrics.csv")

rerank_metrics = rerank_metrics.rename(columns={"label": "split"})

methods_stage2 = {
    "BM25 Dense Fusion": {
        "df": hybrid_stage1,
        "color": "#2ca02c",
        "marker": "D",
    },
    "Rerank": {
        "df": rerank_metrics,
        "color": "#1f77b4",
        "marker": "o",
    },
}

recall_cols_all = []
for cfg in methods_stage2.values():
    cols = [c for c in cfg["df"].columns if c.startswith("MeanR@")]
    recall_cols_all.append(set(cols))

recall_cols_common = sorted(
    set.intersection(*recall_cols_all),
    key=lambda c: int(c.split("@")[1]),
)
k_vals_recall = [int(c.split("@")[1]) for c in recall_cols_common if 10 <= int(c.split("@")[1]) <= 300]
recall_cols_common = [f"MeanR@{k}" for k in k_vals_recall]

print("Stage2+ K values (≤300):", k_vals_recall)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

global_ymin, global_ymax = 1.0, 0.0
for split in splits:
    for cfg in methods_stage2.values():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols_common]
        global_ymin = min(global_ymin, min(vals))
        global_ymax = max(global_ymax, max(vals))

y_pad = (global_ymax - global_ymin) * 0.05
global_ymin = max(0, global_ymin - y_pad)
global_ymax = min(1, global_ymax + y_pad)

tick_candidates_recall = [10, 50, 100, 200, 300]
tick_values_recall = [k for k in tick_candidates_recall if k in k_vals_recall]

for idx, split in enumerate(splits):
    ax = axes[idx]
    for method_name, cfg in methods_stage2.items():
        row = cfg["df"][cfg["df"]["split"] == split]
        if row.empty:
            continue
        vals = [row.iloc[0][c] for c in recall_cols_common]
        ax.plot(
            k_vals_recall,
            vals,
            marker=cfg["marker"],
            color=cfg["color"],
            label=method_name,
            markersize=6,
            linewidth=1.8,
        )

    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(global_ymin, global_ymax)

    if idx == 0:
        ax.set_ylabel("Mean Recall")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("K")
    ax.set_xticks(tick_values_recall)
    ax.set_xticklabels([str(k) for k in tick_values_recall], rotation=90)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.15), fontsize=14)

fig.suptitle("Retrieval vs Rerank Mean_Recall@K (K ≤ 300)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "02_hybrid_rerank_fusion_recall_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 5. Retrieval vs Rerank vs Post-rerank Fusion – MAP@10 (Bar Plots)

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

method_colors_bar = {
    "BM25 Dense Fusion": "#2ca02c",
    "Rerank": "#1f77b4",
    "Post-rerank fusion": "#ff7f0e",
}

for idx, split in enumerate(splits):
    cur_ax = ax[idx]
    vals = []
    labels_methods = []

    row_h = hybrid_stage1[hybrid_stage1["split"] == split]
    if not row_h.empty:
        vals.append(float(row_h.iloc[0]["MAP@10"]))
        labels_methods.append("BM25 Dense Fusion")

    row_r = rerank_metrics[rerank_metrics["split"] == split]
    if not row_r.empty:
        vals.append(float(row_r.iloc[0]["MAP@10"]))
        labels_methods.append("Rerank")

    row_f = rerank_fusion_metrics[rerank_fusion_metrics["split"] == split]
    if not row_f.empty:
        vals.append(float(row_f.iloc[0]["MAP@10"]))
        labels_methods.append("Post-rerank fusion")

    x = np.arange(len(labels_methods))
    colors = [method_colors_bar[label] for label in labels_methods]
    cur_ax.bar(x, vals, color=colors)
    cur_ax.set_xticks(x)
    cur_ax.set_xticklabels("", rotation=30, ha="right")
    cur_ax.set_title(split_labels.get(split, split), fontsize=13, fontweight="bold")

handles_bar = [
    plt.matplotlib.patches.Patch(color=method_colors_bar[m], label=m)
    for m in method_colors_bar
]
fig.legend(handles=handles_bar, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.9])
fig_path = output_dir / "03_hybrid_rerank_fusion_map10_bars_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 6. Retrieval vs Rerank vs Post-rerank Fusion – MAP@K Curves

# %%
map_ks = [1, 3, 5, 10, 20, 30, 40, 50, 75, 100]


def _extract_pmid(doc_entry):
    if isinstance(doc_entry, dict):
        doc_entry = doc_entry.get("document", "")
    if not isinstance(doc_entry, str):
        return None
    if "/" in doc_entry:
        return doc_entry.rsplit("/", 1)[-1]
    return doc_entry


def _load_qrels(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qrels: dict[str, set[str]] = {}
    for q in data.get("questions", []):
        qid = str(q.get("id"))
        docs = q.get("documents", [])
        pmids = {
            _extract_pmid(d)
            for d in docs
            if _extract_pmid(d)
        }
        if qid and pmids:
            qrels[qid] = pmids
    return qrels


def _load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("qid")
    doc_col = cols.get("docno") or cols.get("docid") or cols.get("doc")
    rank_col = cols.get("rank")
    if qid_col is None or doc_col is None:
        raise ValueError(f"Missing qid/doc columns in {path}")
    df[qid_col] = df[qid_col].astype(str)
    df[doc_col] = df[doc_col].astype(str)
    if rank_col:
        df = df.sort_values([qid_col, rank_col])
    return df[[qid_col, doc_col]]


def _ap_at_k(docs: list[str], rels: set[str], k: int) -> float:
    if not rels:
        return 0.0
    denom = min(len(rels), k)
    if denom == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc in enumerate(docs[:k], start=1):
        if doc in rels:
            hits += 1
            score += hits / i
    return score / denom


def _recall_at_k(docs: list[str], rels: set[str], k: int) -> float:
    if not rels:
        return 0.0
    hits = sum(1 for d in docs[:k] if d in rels)
    return hits / len(rels)


def _map_at_ks_for_run(run_df: pd.DataFrame, qrels: dict[str, set[str]], ks: list[int]) -> dict[int, float]:
    qid_col, doc_col = run_df.columns.tolist()
    per_q: dict[int, list[float]] = {k: [] for k in ks}
    for qid, group in run_df.groupby(qid_col, sort=False):
        rels = qrels.get(str(qid))
        if not rels:
            continue
        docs = group[doc_col].tolist()
        for k in ks:
            per_q[k].append(_ap_at_k(docs, rels, k))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in per_q.items()}


qrels_paths = {
    "dicty_gold_llm_public_train_200": base_dir / "example" / "dicty_gold_llm_public_train_200.jsonl",
    "dicty_gold_llm_public_test_50": base_dir / "example" / "dicty_gold_llm_public_test_50.jsonl",
}

qrels_by_split = {s: _load_qrels(p) for s, p in qrels_paths.items()}

run_dirs = {
    "BM25 Dense Fusion": workflow_dir / "hybrid" / "runs",
    "Rerank": workflow_dir / "rerank" / "runs",
    "Post-rerank fusion": workflow_dir / "rerank_hybrid_200" / "runs",
}

map_curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

for split in splits:
    qrels_split = qrels_by_split.get(split, {})
    for method_name, runs_dir in run_dirs.items():
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)
        map_vals = _map_at_ks_for_run(run_df, qrels_split, map_ks)
        map_curves[method_name][split] = map_vals

fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)

all_maps = []
for method_dict in map_curves.values():
    for split_vals in method_dict.values():
        all_maps.extend(split_vals.values())

if all_maps:
    y_min = max(0.0, min(all_maps) - 0.02)
    y_max = min(1.0, max(all_maps) + 0.02)
else:
    y_min, y_max = 0.0, 1.0

colors_map = {
    "BM25 Dense Fusion": "#2ca02c",
    "Rerank": "#1f77b4",
    "Post-rerank fusion": "#ff7f0e",
}

for idx, split in enumerate(splits):
    ax = axes[idx]
    for method_name, method_dict in map_curves.items():
        if split not in method_dict:
            continue
        vals = [method_dict[split].get(k, 0.0) for k in map_ks]
        ax.plot(
            map_ks,
            vals,
            marker="o",
            color=colors_map.get(method_name),
            label=method_name,
            linewidth=1.8,
        )
    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    if idx == 0:
        ax.set_ylabel("MAP@K")
    else:
        ax.set_ylabel("")

    ax.set_xlabel("K")
    ax.set_xticks(map_ks)
    ax.set_xticklabels([str(k) for k in map_ks], rotation=90)
    ax.set_xlim(0, 100)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.15), fontsize=13)

fig.suptitle("Retrieval vs Rerank vs Post Rerank Fusion – MAP@K", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "04_hybrid_rerank_fusion_mapk_per_split.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# Same MAP@K curves without post-rerank fusion (clearer when fusion overlaps / dominates).
map_curves_rr = {k: v for k, v in map_curves.items() if k != "Post-rerank fusion"}
colors_map_rr = {
    "BM25 Dense Fusion": colors_map["BM25 Dense Fusion"],
    "Rerank": "#9467bd",
}
all_maps_rr = []
for method_dict in map_curves_rr.values():
    for split_vals in method_dict.values():
        all_maps_rr.extend(split_vals.values())

if all_maps_rr:
    y_min_rr = max(0.0, min(all_maps_rr) - 0.02)
    y_max_rr = min(1.0, max(all_maps_rr) + 0.02)
else:
    y_min_rr, y_max_rr = 0.0, 1.0

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
for idx, split in enumerate(splits):
    ax = axes2[idx]
    for method_name, method_dict in map_curves_rr.items():
        if split not in method_dict:
            continue
        vals = [method_dict[split].get(k, 0.0) for k in map_ks]
        ax.plot(
            map_ks,
            vals,
            marker="o",
            color=colors_map_rr.get(method_name),
            label=method_name,
            linewidth=1.8,
        )
    ax.set_title(split_labels.get(split, split), fontsize=15, fontweight="bold")
    ax.set_ylim(y_min_rr, y_max_rr)
    if idx == 0:
        ax.set_ylabel("MAP@K")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("K")
    ax.set_xticks(map_ks)
    ax.set_xticklabels([str(k) for k in map_ks], rotation=90)
    ax.set_xlim(0, 100)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")

handles2, labels2 = axes2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc="lower right", bbox_to_anchor=(0.95, 0.15), fontsize=13)
fig2.suptitle("Retrieval vs Rerank – MAP@K", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path_rr = output_dir / "04_hybrid_rerank_mapk_per_split_no_post_fusion.png"
plt.savefig(fig_path_rr, dpi=150, bbox_inches="tight")
print("Saved:", fig_path_rr)
plt.show()

# %% [markdown]
# ## 7. Gold Count Histogram per Query (Dev + Test Combined)

# %%
dev_split = "dicty_gold_llm_public_train_200"
test_split = "dicty_gold_llm_public_test_50"

gold_counts_all = [len(v) for v in qrels_by_split[dev_split].values()]
gold_counts_all.extend(len(v) for v in qrels_by_split[test_split].values())

max_gold = max(gold_counts_all) if gold_counts_all else 0
bins = range(1, max_gold + 2) if max_gold > 0 else [0, 1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(gold_counts_all, bins=bins, color="#4c72b0", alpha=0.8, edgecolor="white")
ax.set_xlabel("|gold| (relevant docs)")
ax.set_ylabel("# queries")
ax.set_title(f"Dev + Test (n={len(gold_counts_all)})", fontweight="bold")

fig.suptitle("Gold Document Count per Query (All Splits)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = output_dir / "05_gold_count_hist_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 8. MAP@K Stratified by Gold Count (All Splits; |gold| = 1, >1)

# %%
bucket_order = ["1", ">1"]


def _gold_bucket(n: int) -> str:
    if n == 1:
        return "1"
    return ">1"


records = []
for method_name, runs_dir in run_dirs.items():
    for split in splits:
        qrels_split = qrels_by_split[split]
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)

        qid_col, doc_col = run_df.columns.tolist()
        for qid, group in run_df.groupby(qid_col, sort=False):
            rels = qrels_split.get(str(qid))
            if not rels:
                continue
            gold_n = len(rels)
            bucket = _gold_bucket(gold_n)
            docs = group[doc_col].tolist()
            for k in map_ks:
                ap = _ap_at_k(docs, set(rels), k)
                records.append(
                    {
                        "split": split,
                        "method": method_name,
                        "qid": str(qid),
                        "gold_n": gold_n,
                        "bucket": bucket,
                        "k": k,
                        "AP@K": ap,
                    }
                )

map_pq_df = pd.DataFrame(records)
print("Per-query AP@K rows:", len(map_pq_df))

summary = (
    map_pq_df.groupby(["method", "bucket", "k"], as_index=False)["AP@K"]
    .mean()
    .rename(columns={"AP@K": "MAP@K"})
)

bucket_counts = (
    map_pq_df[["qid", "bucket"]]
    .drop_duplicates()
    .groupby("bucket")["qid"]
    .nunique()
    .to_dict()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for idx, bucket in enumerate(bucket_order):
    ax = axes[idx]
    bucket_df = summary[summary["bucket"] == bucket]
    if bucket_df.empty:
        ax.set_visible(False)
        continue
    for method_name in run_dirs.keys():
        m_df = bucket_df[bucket_df["method"] == method_name]
        if m_df.empty:
            continue
        ax.plot(
            m_df["k"],
            m_df["MAP@K"],
            marker="o",
            label=method_name,
        )
    n_bucket = bucket_counts.get(bucket, 0)
    ax.set_title(f"|gold| = {bucket}, n={n_bucket}", fontweight="bold")
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx == 0:
        ax.set_ylabel("MAP@K")
    ax.set_xlabel("K")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(run_dirs), bbox_to_anchor=(0.5, 1.05), fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "06_mapk_by_gold_bucket_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %%
recall_records = []
for method_name, runs_dir in run_dirs.items():
    for split in splits:
        qrels_split = qrels_by_split[split]
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)
        qid_col, doc_col = run_df.columns.tolist()
        for qid, group in run_df.groupby(qid_col, sort=False):
            rels = qrels_split.get(str(qid))
            if not rels:
                continue
            docs = group[doc_col].tolist()
            bucket = _gold_bucket(len(rels))
            for k in map_ks:
                recall_records.append({
                    "method": method_name,
                    "bucket": bucket,
                    "k": k,
                    "Recall@K": _recall_at_k(docs, set(rels), k),
                })

recall_pq_df = pd.DataFrame(recall_records)
recall_summary = (
    recall_pq_df.groupby(["method", "bucket", "k"], as_index=False)["Recall@K"]
    .mean()
)

combined = summary.merge(recall_summary, on=["method", "bucket", "k"], how="outer")

all_vals = list(combined["MAP@K"].dropna()) + list(combined["Recall@K"].dropna())
y_min_c = max(0.0, min(all_vals) - 0.02) if all_vals else 0.0
y_max_c = min(1.0, max(all_vals) + 0.02) if all_vals else 1.0

method_colors_combined = {name: f"C{i}" for i, name in enumerate(run_dirs)}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for idx, bucket in enumerate(bucket_order):
    ax = axes[idx]
    bucket_df = combined[combined["bucket"] == bucket]
    if bucket_df.empty:
        ax.set_visible(False)
        continue
    for method_name in run_dirs:
        m_df = bucket_df[bucket_df["method"] == method_name].sort_values("k")
        if m_df.empty:
            continue
        c = method_colors_combined[method_name]
        ax.plot(
            m_df["k"],
            m_df["MAP@K"],
            color=c,
            linewidth=2.0,
        )
        ax.plot(
            m_df["k"],
            m_df["Recall@K"],
            color=c,
            linewidth=2.0,
            linestyle="--",
            alpha=0.6,
        )
    n_bucket = bucket_counts.get(bucket, 0)
    ax.set_title(f"|gold| = {bucket}, n={n_bucket}", fontweight="bold")
    ax.set_ylim(y_min_c, y_max_c)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx == 0:
        ax.set_ylabel("Score")
    ax.set_xlabel("K")

method_handles = [
    Line2D([], [], color=method_colors_combined[m], linewidth=2.0, label=m)
    for m in run_dirs
]
style_handles = [
    Line2D([], [], color="black", linewidth=1.8, linestyle="-", label="MAP@K"),
    Line2D([], [], color="black", linewidth=1.8, linestyle="--", label="Recall@K"),
]
fig.legend(
    handles=method_handles + style_handles,
    loc="upper center",
    ncol=len(run_dirs) + 2,
    bbox_to_anchor=(0.5, 1.05),
    fontsize=12,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "06b_mapk_recall_by_gold_bucket_all.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 10. Snippet Rerank vs Rerank Hybrid 200 – MAP@K Curves

# %%
snippet_run_dirs = {
    "Rerank hybrid 200": workflow_dir / "rerank_hybrid_200" / "runs",
    "Snippet rerank": workflow_dir / "snippet_rerank" / "runs",
}

snippet_map_ks = list(range(10, 101, 10))

snippet_map_curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)

for split in splits:
    qrels_split = qrels_by_split.get(split, {})
    if not qrels_split:
        continue
    for method_name, runs_dir in snippet_run_dirs.items():
        pattern = f"best_rrf_{split}_top5000"
        candidates = list(runs_dir.glob(f"{pattern}*.tsv"))
        if not candidates:
            continue
        run_path = candidates[0]
        run_df = _load_run(run_path)
        map_vals = _map_at_ks_for_run(run_df, qrels_split, snippet_map_ks)
        snippet_map_curves[method_name][split] = map_vals

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

all_vals_snippet: list[float] = []
for method_dict in snippet_map_curves.values():
    for split_vals in method_dict.values():
        all_vals_snippet.extend(split_vals.values())

if all_vals_snippet:
    y_min_s = max(0.0, min(all_vals_snippet) - 0.02)
    y_max_s = min(1.0, max(all_vals_snippet) + 0.02)
else:
    y_min_s, y_max_s = 0.0, 1.0

colors_snippet = {
    "Rerank hybrid 200": "#1f77b4",
    "Snippet rerank": "#ff7f0e",
}

for idx, split in enumerate(splits):
    ax = axes[idx]
    for method_name, method_dict in snippet_map_curves.items():
        if split not in method_dict:
            continue
        vals = [method_dict[split].get(k, 0.0) for k in snippet_map_ks]
        ax.plot(
            snippet_map_ks,
            vals,
            marker="o",
            linewidth=1.8,
            color=colors_snippet[method_name],
            label=method_name,
        )
    ax.set_title(split_labels.get(split, split), fontsize=14, fontweight="bold")
    ax.set_ylim(y_min_s, y_max_s)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x")
    if idx == 0:
        ax.set_ylabel("MAP@K")
    ax.set_xlabel("K")
    ax.set_xticks(snippet_map_ks)
    ax.set_xticklabels([str(k) for k in snippet_map_ks], rotation=90)

legend_snippet_handles = [
    Line2D([0], [0], color=colors_snippet[name], marker="o", linestyle="-", label=name)
    for name in colors_snippet
]
fig.legend(
    handles=legend_snippet_handles,
    labels=list(colors_snippet),
    loc="lower right",
    bbox_to_anchor=(0.95, 0.15),
    fontsize=14,
)
fig.suptitle(
    "Rerank Hybrid 200 vs Snippet Rerank – MAP@K",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = output_dir / "08_snippet_rerank_vs_rerank_hybrid200_mapk.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 11. RRF Fusion Sweep: MAP@10 vs Weight (rerank_hybrid_200 + snippet_rerank)

# %%
RUN_TOP = 100
OUTPUT_TOP = 10
RRF_KS = [60]
RRF_WEIGHTS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]


def _build_run_map(run_df: pd.DataFrame) -> dict[str, list[str]]:
    qid_col, doc_col = run_df.columns.tolist()
    run_map: dict[str, list[str]] = {}
    for qid, group in run_df.groupby(qid_col, sort=False):
        run_map[str(qid)] = group[doc_col].astype(str).tolist()
    return run_map


def _rrf_fuse_docs(
    docs_hybrid: list[str],
    docs_snippet: list[str],
    k_rrf: int,
    w_hybrid: float,
    w_snippet: float,
    run_top: int,
    output_top: int,
) -> list[str]:
    hybrid_top = docs_hybrid[:run_top]
    snippet_top = docs_snippet[:run_top]
    rank_h = {d: i + 1 for i, d in enumerate(hybrid_top)}
    rank_s = {d: i + 1 for i, d in enumerate(snippet_top)}
    union = list(dict.fromkeys(hybrid_top + snippet_top))
    scored: list[tuple[str, float]] = []
    for d in union:
        s = 0.0
        rh = rank_h.get(d)
        rs = rank_s.get(d)
        if rh is not None:
            s += w_hybrid / (k_rrf + rh)
        if rs is not None:
            s += w_snippet / (k_rrf + rs)
        scored.append((d, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored[:output_top]]


def _ap10_for_fusion(
    gold: dict[str, set[str]],
    run_hybrid: dict[str, list[str]],
    run_snippet: dict[str, list[str]],
    k_rrf: int,
    w_hybrid: float,
    w_snippet: float,
) -> tuple[float, int]:
    qids = [q for q in gold if q in run_hybrid and q in run_snippet and gold[q]]
    if not qids:
        return 0.0, 0
    ap_vals: list[float] = []
    for q in qids:
        rels = set(gold[q])
        fused_docs = _rrf_fuse_docs(
            run_hybrid[q],
            run_snippet[q],
            k_rrf=k_rrf,
            w_hybrid=w_hybrid,
            w_snippet=w_snippet,
            run_top=RUN_TOP,
            output_top=OUTPUT_TOP,
        )
        ap_vals.append(_ap_at_k(fused_docs, rels, k=OUTPUT_TOP))
    return (float(np.mean(ap_vals)) if ap_vals else 0.0, len(qids))


hybrid_run_maps: dict[str, dict[str, list[str]]] = {}
snippet_run_maps_sweep: dict[str, dict[str, list[str]]] = {}

for split in splits:
    h_dir = workflow_dir / "rerank_hybrid_200" / "runs"
    h_candidates = list(h_dir.glob(f"best_rrf_{split}_top5000*.tsv"))
    s_dir = workflow_dir / "snippet_rerank" / "runs"
    s_candidates = list(s_dir.glob(f"best_rrf_{split}_top5000*.tsv"))

    if not h_candidates or not s_candidates:
        continue

    df_h = _load_run(h_candidates[0])
    df_s = _load_run(s_candidates[0])

    hybrid_run_maps[split] = _build_run_map(df_h)
    snippet_run_maps_sweep[split] = _build_run_map(df_s)

rrf_rows: list[dict[str, object]] = []

for split in splits:
    gold = qrels_by_split.get(split)
    run_h = hybrid_run_maps.get(split)
    run_s = snippet_run_maps_sweep.get(split)
    if not gold or not run_h or not run_s:
        continue
    for k_rrf in RRF_KS:
        for w_h, w_s in RRF_WEIGHTS:
            map10, n_q = _ap10_for_fusion(
                gold=gold,
                run_hybrid=run_h,
                run_snippet=run_s,
                k_rrf=k_rrf,
                w_hybrid=w_h,
                w_snippet=w_s,
            )
            rrf_rows.append(
                {
                    "split": split,
                    "k_rrf": k_rrf,
                    "w_hybrid": w_h,
                    "w_snippet": w_s,
                    "MAP@10": map10,
                    "n_queries": n_q,
                }
            )

rrf_results = pd.DataFrame(rrf_rows)
if not rrf_results.empty:
    rrf_results["weight_label"] = rrf_results.apply(
        lambda r: f"({r['w_hybrid']:.1f},{r['w_snippet']:.1f})",
        axis=1,
    )

weight_order = [f"({w[0]:.1f},{w[1]:.1f})" for w in RRF_WEIGHTS]

if not rrf_results.empty:
    n_splits = rrf_results["split"].nunique()
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 4), sharey=True)
    if n_splits == 1:
        axes = np.array([axes])
    axes_flat = list(axes.flat)

    for idx, (ax, (split, grp)) in enumerate(zip(axes_flat, rrf_results.groupby("split", sort=False))):
        for k_rrf in sorted(grp["k_rrf"].unique()):
            sub = grp[grp["k_rrf"] == k_rrf].set_index("weight_label").reindex(weight_order)
            vals = sub["MAP@10"].values
            ax.plot(
                range(len(weight_order)),
                vals,
                marker="o",
                linewidth=1.6,
            )
        ax.set_xticks(range(len(weight_order)))
        ax.set_xticklabels(weight_order, rotation=45, ha="right")
        n_q = int(grp["n_queries"].max())
        ax.set_title(f"{split_labels.get(split, split)} (n={n_q})", fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("MAP@10")
        ax.set_xlabel("(w_hybrid, w_snippet)")
        ax.grid(True, axis="y")

    fig.suptitle("Docs and Snippet Fusion", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = output_dir / "09_rrf_fusion_map10_vs_weight.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("Saved:", fig_path)
plt.show()

# %% [markdown]
# ## 12. Post-rerank Fusion – MAP@K and Recall@K by Evidence Level
#

# %%
EVIDENCE_LEVEL_ORDER = [
    "abstract_support_details",
    "supports_core",
    "needs_fulltext",
]

EVIDENCE_LEVEL_LABELS = {
    "abstract_support_details": "Abstract support details",
    "supports_core": "Supports core",
    "needs_fulltext": "Needs full text",
}

ks_evidence = [1, 5, 10, 20, 30,  50, 75, 100]


def _load_evidence_levels(gold_json: Path) -> dict[str, str]:
    with gold_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qid_to_level: dict[str, str] = {}
    for i, q in enumerate(data.get("questions", [])):
        qid = str(q.get("id") or q.get("qid") or i)
        docs = q.get("docs") or []
        if docs:
            level = (docs[0].get("evidence_level") or "unknown").strip() or "unknown"
        else:
            level = "unknown"
        qid_to_level[qid] = level
    return qid_to_level


for split in splits:
    print(f"Building evidence-level curves for split={split} ...")
    qrels_split = qrels_by_split.get(split, {})
    if not qrels_split:
        print("  No qrels for this split; skipping.")
        continue

    gold_json = qrels_paths[split]
    qid_to_level = _load_evidence_levels(gold_json)

    fusion_runs_dir = workflow_dir / "rerank_hybrid_200" / "runs"
    fusion_candidates = list(
        fusion_runs_dir.glob(f"best_rrf_{split}_top5000_rrf_poolR200_poolH200_k60.tsv")
    )
    if not fusion_candidates:
        print("  No post-rerank fusion run found; skipping.")
        continue
    fusion_run_df = _load_run(fusion_candidates[0])
    qid_col, doc_col = fusion_run_df.columns.tolist()

    run_docs: dict[str, list[str]] = {}
    for qid, group in fusion_run_df.groupby(qid_col, sort=False):
        run_docs[str(qid)] = group[doc_col].tolist()

    levels_present = sorted(
        {lvl for qid, lvl in qid_to_level.items() if qid in run_docs and qid in qrels_split}
    )
    levels = [lvl for lvl in EVIDENCE_LEVEL_ORDER if lvl in levels_present]
    levels += [lvl for lvl in levels_present if lvl not in levels]
    if not levels:
        print("  No overlapping qids with evidence_level; skipping.")
        continue

    map_by_level: dict[str, list[float]] = {lvl: [] for lvl in levels}
    rec_by_level: dict[str, list[float]] = {lvl: [] for lvl in levels}
    n_by_level: dict[str, int] = {lvl: 0 for lvl in levels}

    for lvl in levels:
        qids_lvl = [
            qid
            for qid, l in qid_to_level.items()
            if l == lvl and qid in run_docs and qid in qrels_split
        ]
        if not qids_lvl:
            map_by_level[lvl] = [0.0] * len(ks_evidence)
            rec_by_level[lvl] = [0.0] * len(ks_evidence)
            continue
        n_by_level[lvl] = len(qids_lvl)
        map_vals: list[float] = []
        rec_vals: list[float] = []
        for k in ks_evidence:
            ap_scores: list[float] = []
            rec_scores: list[float] = []
            for qid in qids_lvl:
                rels = qrels_split.get(qid)
                if not rels:
                    continue
                docs = run_docs[qid]
                rels_set = set(rels)
                ap_scores.append(_ap_at_k(docs, rels_set, k))
                rec_scores.append(_recall_at_k(docs, rels_set, k))
            map_vals.append(float(np.mean(ap_scores)) if ap_scores else 0.0)
            rec_vals.append(float(np.mean(rec_scores)) if rec_scores else 0.0)
        map_by_level[lvl] = map_vals
        rec_by_level[lvl] = rec_vals

    all_map_vals = [v for vals in map_by_level.values() for v in vals]
    all_rec_vals = [v for vals in rec_by_level.values() for v in vals]
    y_min_m = max(0.0, min(all_map_vals) - 0.02) if all_map_vals else 0.0
    y_max_m = min(1.0, max(all_map_vals) + 0.02) if all_map_vals else 1.0
    y_min_r = max(0.0, min(all_rec_vals) - 0.02) if all_rec_vals else 0.0
    y_max_r = min(1.0, max(all_rec_vals) + 0.02) if all_rec_vals else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes_flat = axes.flatten()

    for idx, lvl in enumerate(levels):
        col_title = EVIDENCE_LEVEL_LABELS.get(lvl, lvl)
        n_q = n_by_level.get(lvl, 0)
        ax_map = axes_flat[idx]
        ax_map.plot(
            ks_evidence,
            map_by_level[lvl],
            marker="o",
            color="#1f77b4",
            linewidth=1.8,
        )
        ax_map.set_title(f"{col_title} (n={n_q})", fontweight="bold")
        ax_map.set_ylim(y_min_m, y_max_m)
        ax_map.set_ylabel("MAP@K" if idx == 0 else "")
        if idx != 0:
            ax_map.tick_params(axis="y", labelleft=False)
        ax_map.grid(True, axis="y")
        ax_map.grid(True, axis="x")

        ax_rec = axes_flat[idx + 3]
        ax_rec.plot(
            ks_evidence,
            rec_by_level[lvl],
            marker="o",
            color="#ff7f0e",
            linewidth=1.8,
        )
        ax_rec.set_ylim(y_min_r, y_max_r)
        ax_rec.set_ylabel("Recall@K" if idx == 0 else "")
        if idx != 0:
            ax_rec.tick_params(axis="y", labelleft=False)
        ax_rec.set_xlabel("K")
        ax_rec.grid(True, axis="y")
        ax_rec.grid(True, axis="x")

    for j in range(len(levels), 3):
        axes_flat[j].set_visible(False)
        axes_flat[j + 3].set_visible(False)

    for ax in axes_flat:
        ax.set_xlim(0, 100)

    fig.suptitle(
        f"Post-rerank Fusion – MAP@K and Recall@K by Evidence Level ({split_labels.get(split, split)})",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = (
        output_dir
        / f"12_post_rerank_fusion_map_recall_by_evidence_level_{split_labels.get(split, split)}.png"
    )
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print("Saved:", fig_path)
    plt.show()

# %%

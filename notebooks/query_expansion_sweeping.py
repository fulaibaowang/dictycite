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

# %%
import os
from pathlib import Path
import pandas as pd
import polars as pl

# %% [markdown]
# ## Query-field sweep: statistical comparison (recall 200–1000)
#
# We compare retrieval profiles (e.g. hybrid combo bm25_qf × dense_qf) using **mean recall over K ∈ {200, 300, 400, 500, 1000}** so the choice of operating point (e.g. rerank cutoff) does not dominate. Data from `scripts/public/combine_query_field_sweep_results.py` (combined sweep metrics). Stats: mean ± std across batches (train/test), rank, and optional paired test of best vs others.

# %%
# Path to combined sweep metrics (from combine_query_field_sweep_results.py)

SWEEP_CSV = Path(os.environ.get("DICTYCITE_ROOT", "../")) / "output/workflow_hpc_test/combined_sweep_metrics.csv"
if not SWEEP_CSV.is_file():
    SWEEP_CSV = Path("../output/workflow_hpc_test/combined_sweep_metrics.csv")

K_RANGE = [200, 300, 400, 500, 1000]  # recall@K range for "operating region" metric
MEANR_COLS = [f"MeanR@{k}" for k in K_RANGE]

df = pd.read_csv(SWEEP_CSV)
# Ensure we have the recall columns (hybrid/bm25/dense may have different column sets)
available = [c for c in MEANR_COLS if c in df.columns]
if not available:
    raise ValueError(f"None of {MEANR_COLS} found in {SWEEP_CSV}. Columns: {list(df.columns)}")

# Mean recall over K in [200, 500, 1000] per (profile, batch). Profile = stage + query fields.
df["batch"] = df["batch"].astype(str).str.strip()
if "stage" in df.columns:
    df["stage"] = df["stage"].astype(str).str.strip()
if "bm25_query_field" in df.columns and "dense_query_field" in df.columns:
    df["profile"] = df["stage"] + " (" + df["bm25_query_field"].astype(str) + ", " + df["dense_query_field"].astype(str) + ")"
else:
    df["profile"] = df.get("stage", df.get("method", "unknown"))

df["mean_recall_200_1000"] = df[available].mean(axis=1)

# Per-profile summary: mean and std across batches
summary = df.groupby("profile").agg(
    mean_recall=("mean_recall_200_1000", "mean"),
    std_recall=("mean_recall_200_1000", "std"),
    n_batches=("batch", "nunique"),
).reset_index()
summary = summary.sort_values("mean_recall", ascending=False).reset_index(drop=True)
summary["rank"] = range(1, len(summary) + 1)

print("Recall (mean over K in {200..1000}), mean ± std across batches:\n")
display(pl.from_pandas(summary))

# Paired comparison: best profile vs others (using the 5 K-values as repeated measures per batch)
try:
    import scipy.stats as st
except ImportError:
    st = None

best_profile = summary.iloc[0]["profile"]
if st and len(summary) > 1 and len(df["batch"].unique()) >= 2:
    # For each batch we have one value per profile. Compare best vs each other.
    piv = df.pivot_table(index="batch", columns="profile", values="mean_recall_200_1000")
    if best_profile in piv.columns:
        best_vals = piv[best_profile].dropna()
        for other in summary.iloc[1:]["profile"]:
            if other not in piv.columns:
                continue
            other_vals = piv[other].dropna()
            common = best_vals.index.intersection(other_vals.index)
            if len(common) < 2:
                continue
            t, p = st.ttest_rel(best_vals.loc[common], other_vals.loc[common])
            print(f"\nPaired t-test (best vs {other}): t={t:.4f}, p={p:.4f}")
else:
    print("\n(Paired test skipped: need scipy and ≥2 batches)")


# %% [markdown]
# ## Rerank comparison: MAP@10 by query field (body vs synonyms vs long)
#
# After reranking, we compare **MAP@10** across methods (body, synonyms, long, hybrid) using the output of `scripts/public/shared_scripts/compare_result_dirs.py` (compare_summary.csv). Stats: mean ± std across train/test splits, rank, and paired t-test of best vs others. This complements the retrieval recall comparison above: expansion helps recall at retrieval but does not improve MAP after reranking.

# %%
# Path to compare summary (from compare_result_dirs.py --output-dir .../compare_plots)
import os
from pathlib import Path

COMPARE_CSV = Path(os.environ.get("DICTYCITE_ROOT", "../")) / "output/workflow_hpc_test/fixed_long_rerank_sweep/compare_plots/compare_summary.csv"
if not COMPARE_CSV.is_file():
    COMPARE_CSV = Path("../output/workflow_hpc_test/fixed_long_rerank_sweep/compare_plots/compare_summary.csv")

if not COMPARE_CSV.is_file():
    print("Compare summary not found. Run compare_result_dirs.py with --output-dir .../compare_plots first.")
else:
    df = pd.read_csv(COMPARE_CSV)
    if "MAP@10" not in df.columns:
        print("MAP@10 not in compare summary. Columns:", list(df.columns))
    else:
        # One row per (dir_label, role). Aggregate by dir_label.
        summary = df.groupby("dir_label").agg(
            mean_map10=("MAP@10", "mean"),
            std_map10=("MAP@10", "std"),
            n_splits=("role", "nunique"),
        ).reset_index()
        summary = summary.sort_values("mean_map10", ascending=False).reset_index(drop=True)
        summary["rank"] = range(1, len(summary) + 1)

        print("MAP@10 after reranking, mean ± std across splits (train/test):\n")
        display(pl.from_pandas(summary))

        # Paired t-test: best vs others (by role/split)
        try:
            import scipy.stats as st
        except ImportError:
            st = None

        best_method = summary.iloc[0]["dir_label"]
        if st and len(summary) > 1 and "role" in df.columns and df["role"].nunique() >= 2:
            piv = df.pivot_table(index="role", columns="dir_label", values="MAP@10")
            if best_method in piv.columns:
                best_vals = piv[best_method].dropna()
                for other in summary.iloc[1:]["dir_label"]:
                    if other not in piv.columns:
                        continue
                    other_vals = piv[other].dropna()
                    common = best_vals.index.intersection(other_vals.index)
                    if len(common) < 2:
                        continue
                    t, p = st.ttest_rel(best_vals.loc[common], other_vals.loc[common])
                    print(f"\nPaired t-test (best vs {other}): t={t:.4f}, p={p:.4f}")
        else:
            print("\n(Paired test skipped: need scipy and ≥2 splits in compare summary)")

# %%

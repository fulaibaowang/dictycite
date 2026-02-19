# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Query expansion and BM25 test
#
# This notebook builds **two expansion variants** for the goldset and runs BM25 (and RM3) retrieval tests.
#
# **Inputs**
# - Gold parquet (e.g. `output/cleaned/golden_grouped.parquet`) with columns `query`, `docs`, etc.
# - dictybase_files/gene_information.txt: tab-separated table with columns GENE ID, Gene Name, Synonyms (comma-separated), Gene products. Used only for query expansion; ambiguous aliases (mapping to multiple genes) are skipped.
#
# **Expansion rules**
# - Detection: A query is expanded when it contains (1) a token matching DDB_G\d+, or (2) a token matching a gene name/synonym (case-insensitive) from the table.
# - Expansion: For each detected gene we append a bounded list of aliases (gene name + synonyms). We never append gene IDs (DDB_G...) even when the gene was detected via ID.
# - Two variants: query_expand_synonyms (id, gene name, synonyms only) and query_expand_long (synonyms + gene products).
#
# **Output**: Parquet with query, query_expand_synonyms, query_expand_long, query_expand (= long), docs; flat TSV for labeling.

# %%
import polars as pl
import pandas as pd
import pyterrier as pt
import numpy as np

import os, math
import matplotlib.pyplot as plt
import re


# %%
if not pt.started():
    pt.init()

# %% [markdown]
# # let us do one query test

# %%
INDEX_PATH = os.path.abspath("../indexes/terrier_index_dicty_22.01.26")

# 1) load existing index from disk
index_ref = pt.IndexRef.of(INDEX_PATH)
index = pt.IndexFactory.of(index_ref)

# 2) BM25 retriever
br = pt.BatchRetrieve(index, wmodel="BM25")


# %%
# 3) pick one query from gold (change row index if you want)
gold = pl.read_parquet("../output/cleaned/golden_grouped.parquet")

row = gold.select(["group_claim_id", "query", "docs"]).row(100, named=True)


# %%
qid = str(row["group_claim_id"])
query = row["query"]
# positives = pmids from gold docs
docs = row["docs"]  # list of structs
pos_pmids = sorted({str(d.get("pmid")) for d in docs if d.get("pmid") not in (None, "", "NA")})
pos_set = set(pos_pmids)

# %%
print("qid:", qid)
print("query:", query)
print("n_pos_pmids:", len(pos_pmids))
print("pos_pmids (first 20):", pos_pmids[:20])


# %%
# 4) retrieve top K
K = 1000
qdf = pd.DataFrame([{"qid": qid, "query": query}])
res = br.transform(qdf).head(K)
res["docno"] = res["docno"].astype(str)

# 5) check where positives appear
hits = res[res["docno"].isin(pos_set)].copy().sort_values("rank")
missed = sorted(pos_set - set(res["docno"]))

print("\n--- Positive hits in top", K, "---")
print("n_hits:", len(hits), "/", len(pos_pmids))
print("best_rank:", (int(hits["rank"].min()) if len(hits) else None))
print("missed_in_topK:", len(missed))
print("missed pmids (first 20):", missed[:20])

display(hits[["qid","docno","rank","score"]].head(50))

for k in [10, 20, 50, 100, 200, 500, 1000]:
    topk = set(res[res["rank"] < k]["docno"])
    rec = len(topk & pos_set) / (len(pos_set) if len(pos_set) else 1)
    print(f"recall@{k}: {rec:.3f}")

# %%
# # check the abstract
# gid = gold.select("group_claim_id").row(0)[0]   # scalar int
# # or: gid = gold.get_column("group_claim_id")[0]

# pos_docs = (
#     gold
#     .filter(pl.col("group_claim_id") == gid)
#     .select(["group_claim_id", "query", "docs"])
#     .explode("docs")
#     .with_columns([
#         pl.col("docs").struct.field("pmid").alias("pmid"),
#         pl.col("docs").struct.field("publication_id").alias("publication_id"),
#         pl.col("docs").struct.field("title").alias("title"),
#         pl.col("docs").struct.field("abstract_clean").alias("abstract_clean"),
#         pl.col("docs").struct.field("year").alias("year"),
#     ])
#     .drop("docs")
# )

# with pl.Config(fmt_str_lengths=10_000, tbl_rows=200, tbl_cols=20):
#     display(pos_docs)

# %% [markdown]
# In this first row, BM25 gives the citation at rank 0, so perfectly matched

# %% [markdown]
# # let us prepare the results for gold and BM25

# %%
# -------------------------
# build mappings for gold
# -------------------------

# qid -> query
queries_df = gold.select([
    pl.col("group_claim_id").cast(pl.Utf8).alias("qid"),
    pl.col("query").alias("query"),
])

# qid -> set(pmids)
gold_pmids = (
    gold.select(["group_claim_id", "docs"])
        .explode("docs")
        .with_columns([
            pl.col("group_claim_id").cast(pl.Utf8).alias("qid"),
            pl.col("docs").struct.field("pmid").cast(pl.Utf8).alias("pmid"),
        ])
        .filter(pl.col("pmid").is_not_null() & (pl.col("pmid") != "") & (pl.col("pmid") != "NA"))
        .select(["qid", "pmid"])
        .unique()
)

gold_map: dict[str, set[str]] = {}
for qid, pmid in gold_pmids.iter_rows():
    gold_map.setdefault(qid, set()).add(pmid)


# %%
# -------------------------
# Run retrieval (overfetch)
# -------------------------
qdf = queries_df.to_pandas()

K_MAX = 500
res_all = br.transform(qdf)

res = (
    res_all
    .sort_values(["qid", "rank"], ascending=[True, True])
    .groupby("qid", as_index=False)
    .head(K_MAX)
)
res["qid"] = res["qid"].astype(str)
res["docno"] = res["docno"].astype(str)
res = res.sort_values(["qid", "rank"], ascending=[True, True])

# Build run_map: qid -> ranked list (docno)
run_map: dict[str, list[str]] = {}
for qid, grp in res.groupby("qid", sort=False):
    run_map[qid] = grp["docno"].tolist()

# %%
res_all.shape, res.shape

# %%
len(run_map)


# %% [markdown]
# # let us implement the eval strategy from BIOASQ

# %%
# -------------------------
# BioASQ-style AP + extras
# -------------------------
def ap_bioasq(ranked: list[str], relset: set[str], k: int = 10) -> float:
    """
    AP with denominator min(|relset|, k), matching BioASQ Phase A doc AP behavior.
    """
    ranked = ranked[:k]
    if not relset:
        return 0.0
    denom = min(len(relset), k)
    if denom == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, docno in enumerate(ranked, start=1):
        if docno in relset:
            hits += 1
            s += hits / i
    return s / denom

def rr_at_k(ranked: list[str], relset: set[str], k: int = 10) -> float:
    ranked = ranked[:k]
    for i, docno in enumerate(ranked, start=1):
        if docno in relset:
            return 1.0 / i
    return 0.0

def success_at_k(ranked: list[str], relset: set[str], k: int = 10) -> int:
    ranked = ranked[:k]
    return int(any(docno in relset for docno in ranked))

def recall_at_k(ranked: list[str], relset: set[str], k: int) -> float:
    ranked = ranked[:k]
    if not relset:
        return 0.0
    return len(set(ranked) & relset) / len(relset)


# %%
# -------------------------
# Evaluate baseline BM25
# -------------------------
Ks_recall = [50, 100, 200, 500]
eps_power = 5   # matches BioASQ -e 5 (epsilon = 1e-5)
eps = 10 ** (-eps_power)

def evaluate_run(gold_map: dict[str, set[str]], run_map: dict[str, list[str]], k_max: int, ks_recall: list[int] = None) -> tuple[dict, pl.DataFrame]:
    """Generic evaluation function for any run_map against gold_map"""
    if ks_recall is None:
        ks_recall = [50, 100, 200, 500]
    
    perq = []
    APs, RRs, S10s = [], [], []
    recalls = {K: [] for K in ks_recall}

    for qid, relset in gold_map.items():
        ranked = run_map.get(qid, [])

        ap = ap_bioasq(ranked, relset, k=10)
        rr = rr_at_k(ranked, relset, k=10)
        s10 = success_at_k(ranked, relset, k=10)

        APs.append(ap)
        RRs.append(rr)
        S10s.append(s10)

        rec_k_vals = {}
        for K in ks_recall:
            rK = recall_at_k(ranked, relset, k=K)
            recalls[K].append(rK)
            rec_k_vals[f"recall@{K}"] = rK

        perq.append({
            "qid": qid,
            "n_gold": len(relset),
            "AP@10": ap,
            "RR@10": rr,
            "Success@10": s10,
            **rec_k_vals,
        })

    MAP10 = sum(APs) / len(APs) if APs else 0.0
    GMAP10 = math.exp(sum(math.log(a + eps) for a in APs) / len(APs)) if APs else 0.0
    MRR10 = sum(RRs) / len(RRs) if RRs else 0.0
    Success10 = sum(S10s) / len(S10s) if S10s else 0.0
    RecallK = {K: (sum(vals) / len(vals) if vals else 0.0) for K, vals in recalls.items()}

    summary = {
        "MRR@10": MRR10,
        "Success@10": Success10,
        "MAP@10": MAP10,
        "GMAP@10": GMAP10,
        **{f"Recall@{K}": RecallK[K] for K in ks_recall},
        "n_queries": len(gold_map),
        "K_MAX_retrieved": k_max,
    }
    
    perq_df = pl.DataFrame(perq).sort("RR@10")
    return summary, perq_df

def run_retrieval(retriever, qdf: pd.DataFrame, k_max: int) -> dict[str, list[str]]:
    """Run retrieval and build run_map: qid -> ranked docno list"""
    res_all = retriever.transform(qdf)
    res = (
        res_all
        .sort_values(["qid", "rank"], ascending=[True, True])
        .groupby("qid", as_index=False)
        .head(k_max)
    )
    res["qid"] = res["qid"].astype(str)
    res["docno"] = res["docno"].astype(str)
    
    run_map = {}
    for qid, grp in res.groupby("qid", sort=False):
        run_map[qid] = grp["docno"].tolist()
    return run_map

summary, perq_df = evaluate_run(gold_map, run_map, K_MAX, Ks_recall)
print(summary)
perq_df

# %%
# perq_df: Polars DataFrame from evaluation (columns: "RR@10","AP@10","Success@10","recall@50"...)
# summary: dict from evaluation (keys: "MRR@10", "Recall@50", ...)

pdf = perq_df.to_pandas()

# 1) RR@10 distribution
plt.figure()
plt.hist(pdf["RR@10"], bins=50)
plt.title("Per-query RR@10 distribution")
plt.xlabel("RR@10")
plt.ylabel("Number of queries")
plt.show()

# 2) AP@10 distribution
plt.figure()
plt.hist(pdf["AP@10"], bins=50)
plt.title("Per-query AP@10 distribution")
plt.xlabel("AP@10")
plt.ylabel("Number of queries")
plt.show()

# 3) Success@10 distribution (0/1)
plt.figure()
plt.hist(pdf["Success@10"], bins=[-0.5, 0.5, 1.5])
plt.title("Per-query Success@10 distribution")
plt.xlabel("Success@10")
plt.ylabel("Number of queries")
plt.xticks([0, 1])
plt.show()

# %% [markdown]
# this can be seen as a baseline result. looks OK

# %%
# 4) Mean Recall@K bar chart
Ks = [50, 100, 200, 500]
rec_vals = [summary.get(f"Recall@{k}", 0.0) for k in Ks]

plt.figure()
plt.bar([str(k) for k in Ks], rec_vals)
plt.title("Mean Recall@K (BM25 candidate set coverage)")
plt.xlabel("K")
plt.ylabel("Mean Recall@K")
plt.ylim(0, 1)
plt.show()

# 5) Optional: per-query Recall@K distributions
for k in Ks:
    col = f"recall@{k}"
    if col in pdf.columns:
        plt.figure()
        plt.hist(pdf[col], bins=30)
        plt.title(f"Per-query {col} distribution")
        plt.xlabel(col)
        plt.ylabel("Number of queries")
        plt.ylim(bottom=0)
        plt.show()

# %%
# qids where BM25 retrieved none of the gold docs in top 500
zero_qids = (
    perq_df
    .filter(pl.col("recall@500") == 0)
    .select(pl.col("qid").cast(pl.Utf8))
)

# Pull query + gold docs for those qids
zero_cases = (
    gold
    .with_columns(pl.col("group_claim_id").cast(pl.Utf8).alias("qid"))
    .join(zero_qids, on="qid", how="inner")
    .select(["qid", "query", "docs", "n_citations", "n_variants"])
    .explode("docs")
    .with_columns([
        pl.col("docs").struct.field("pmid").alias("pmid"),
        pl.col("docs").struct.field("publication_id").alias("publication_id"),
        pl.col("docs").struct.field("title").alias("title"),
        pl.col("docs").struct.field("abstract_clean").alias("abstract_clean"),
        pl.col("docs").struct.field("year").alias("year"),
    ])
    .drop("docs")
    .sort(["qid", "pmid"])
)

# with pl.Config(fmt_str_lengths=10_000, tbl_rows=5000, tbl_cols=30):
#     display(zero_cases)
zero_cases

# %%
with pl.Config(fmt_str_lengths=10_000, tbl_rows=5, tbl_cols=30):
    display(zero_cases)


# %%
zero_cases.write_csv("../output/cleaned/golden_zero_recall500_to_inspect.tsv", separator="\t")


# %% [markdown]
# just from looking at this, it seems 0 at recall@500 is because this claims are summarized from text not abstract. so missing them is as expected

# %% [markdown]
# # tune K1 and b

# %%
# -----------------------------
# Settings
# -----------------------------
K_CAND = 200          # retrieval depth for reranker candidate set
K_RR = 10             # MRR@10
DEV_FRAC = 0.8        # fraction split dev/test
SEED = 1

# Grid (feel free to edit)
k1_list = [0.6, 0.9, 1.2, 1.5, 2.0]
b_list  = [0.3, 0.6, 0.75, 0.9]

# %%
# -----------------------------
# Split dev/test by qid
# -----------------------------
# queries_df is your Polars DF with columns: qid, query
qids_all = queries_df.select(pl.col("qid").cast(pl.Utf8)).get_column("qid").to_list()

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(qids_all))
cut = int(len(qids_all) * DEV_FRAC)

dev_qids = set([qids_all[i] for i in perm[:cut]])
test_qids = set([qids_all[i] for i in perm[cut:]])

dev_topics = queries_df.filter(pl.col("qid").is_in(list(dev_qids))).to_pandas()
test_topics = queries_df.filter(pl.col("qid").is_in(list(test_qids))).to_pandas()

dev_gold = {qid: gold_map[qid] for qid in dev_qids if qid in gold_map}
test_gold = {qid: gold_map[qid] for qid in test_qids if qid in gold_map}

print("n_all:", len(qids_all), "n_dev:", len(dev_gold), "n_test:", len(test_gold))


# %%
# -------------------------
# Helpers for grid search
# Note: rr_at_k, recall_at_k, ap_bioasq, success_at_k already defined earlier
# -------------------------
def eval_run(gold_map: dict[str, set[str]], run_map: dict[str, list[str]], k_rr: int = 10, k_rec: int = 200):
    """Quick eval for grid search: returns only MRR@k_rr and Recall@k_rec"""
    qids = list(gold_map.keys())
    rrs = []
    recs = []
    for qid in qids:
        ranked = run_map.get(qid, [])
        relset = gold_map[qid]
        rrs.append(rr_at_k(ranked, relset, k=k_rr))
        recs.append(recall_at_k(ranked, relset, k=k_rec))
    return {
        "MRR@10": float(np.mean(rrs)) if rrs else 0.0,
        f"Recall@{k_rec}": float(np.mean(recs)) if recs else 0.0,
    }

def run_bm25(index, topics_pd: pd.DataFrame, k1: float, b: float, K: int = 200) -> dict[str, list[str]]:
    """Run BM25 with custom k1, b parameters (for grid search)"""
    rtr = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        controls={"bm25.k_1": k1, "bm25.b": b},
        num_results=K
    )
    return run_retrieval(rtr, topics_pd, K)


# %%
# -----------------------------
# Grid search
# -----------------------------
rows = []

for k1 in k1_list:
    for b in b_list:
        run_dev = run_bm25(index, dev_topics, k1=k1, b=b, K=K_CAND)
        dev_metrics = eval_run(dev_gold, run_dev, k_rr=K_RR, k_rec=K_CAND)

        run_test = run_bm25(index, test_topics, k1=k1, b=b, K=K_CAND)
        test_metrics = eval_run(test_gold, run_test, k_rr=K_RR, k_rec=K_CAND)

        # Pick a dev selection criterion: prioritize Recall@200, tie-break by MRR@10
        dev_score = dev_metrics[f"Recall@{K_CAND}"] + 0.1 * dev_metrics["MRR@10"]  # <-------- this can need a second thought

        rows.append({
            "k1": k1,
            "b": b,
            "dev_Recall@200": dev_metrics[f"Recall@{K_CAND}"],
            "dev_MRR@10": dev_metrics["MRR@10"],
            "dev_score": dev_score,
            "test_Recall@200": test_metrics[f"Recall@{K_CAND}"],
            "test_MRR@10": test_metrics["MRR@10"],
        })

results = pl.DataFrame(rows).sort("dev_score", descending=True)

with pl.Config(tbl_rows=200, tbl_cols=20):
    display(results)

best = results.row(0, named=True)
print("\nBEST (by dev_score):", best)



# %%
# save results (optional)
# results.write_csv("../tmp/bm25_gridsearch_devtest.tsv", separator="\t")

# %% [markdown]
# Tune parameters change results unsignificant. Default BM25 (k1=1.2, b=0.75))for the baseline report.
#
# Dev: Recall@200 = 0.886311, MRR@10 = 0.601671
#
# Test: Recall@200 = 0.850140, MRR@10 = 0.589982
#
# Better choices for scoring function:
# pick highest Recall@200
# tie-break by MRR@10

# %% [markdown]
# # look into zero_cases of recall@500
# there are several options for the next step but let us do more research on zero_cases

# %%
zeros = (
    perq_df.filter(pl.col("recall@500") == 0)
           .select("qid")
           .to_series()
           .to_list()
)

ones = (
    perq_df.filter(pl.col("recall@500") == 1)
           .select("qid")
           .to_series()
           .to_list()
)

print("n recall@500==0:", len(zeros))
print("n recall@500==1:", len(ones))


# %%
def show_gold_docs(gold: pl.DataFrame, qid: str):
    g = (
        gold.filter(pl.col("group_claim_id").cast(pl.Utf8) == str(qid))
            .select(["group_claim_id", "query", "docs"])
            .explode("docs")
            .with_columns([
                pl.col("docs").struct.field("pmid").alias("pmid"),
                pl.col("docs").struct.field("title").alias("title"),
                pl.col("docs").struct.field("abstract_clean").alias("abstract_clean"),
                pl.col("docs").struct.field("year").alias("year"),
            ])
            .drop("docs")
    )
    with pl.Config(fmt_str_lengths=10_000, tbl_rows=200, tbl_cols=20):
        display(g)



# %%

def inspect_query(res: pd.DataFrame, gold_map: dict[str, set[str]], gold: pl.DataFrame, qid: str, top_show: int = 20):
    qid = str(qid)
    rel = gold_map.get(qid, set())

    # show query text
    qtxt = gold.filter(pl.col("group_claim_id").cast(pl.Utf8) == qid).select("query").row(0)[0]
    print("\n==============================")
    print("QID:", qid)
    print("QUERY:", qtxt)
    print("N gold pmids:", len(rel))

    # show gold docs
    print("\n--- GOLD DOCS (title/abstract) ---")
    show_gold_docs(gold, qid)

    # get retrieved ranking
    r = res[res["qid"].astype(str) == qid].sort_values("rank").copy()
    r["docno"] = r["docno"].astype(str)

    # compute first relevant rank + hits
    hits = r[r["docno"].isin(rel)].copy().sort_values("rank")
    first_rank = int(hits["rank"].iloc[0]) if len(hits) else None

    print("\n--- BM25 TOP", top_show, "---")
    display(r.head(top_show)[["qid", "docno", "rank", "score"]])

    print("\n--- POSITIVES FOUND IN TOP500 ---")
    print("n_hits:", len(hits), "/", len(rel))
    print("first_relevant_rank:", first_rank)
    if len(hits):
        display(hits.head(50)[["qid", "docno", "rank", "score"]])


# %%
# pick a few examples
for qid in zeros[:5]:
    inspect_query(res, gold_map, gold, qid, top_show=10)


# %%
for qid in ones[:3]:
    inspect_query(res, gold_map, gold, qid, top_show=5)

# %% [markdown]
# # try rm3
# RM3 is a simple method expand the vocabulary of query
# BM25 (initial) → RM3 (query rewrite) → BM25 (final)

# %%
rm3 = pt.rewrite.RM3(index, fb_docs=10, fb_terms=10, fb_lambda=0.6)
pipe_rm3 = br >> rm3 >> br


# %%
res_all_rm3 = pipe_rm3.transform(qdf)

res_rm3 = (
    res_all_rm3
    .sort_values(["qid", "rank"], ascending=[True, True])
    .groupby("qid", as_index=False)
    .head(K_MAX)
)
res_rm3["qid"] = res_rm3["qid"].astype(str)
res_rm3["docno"] = res_rm3["docno"].astype(str)
res_rm3 = res_rm3.sort_values(["qid", "rank"], ascending=[True, True])


# %%
# build run map: qid -> ranked docno list
run_map_rm3 = {}
for qid, grp in res_rm3.groupby("qid", sort=False):
    run_map_rm3[qid] = grp["docno"].tolist()


# %%

# --- evaluate ---
perq = []
APs, RRs, S10s = [], [], []
recalls = {K: [] for K in Ks_recall}

for qid, relset in gold_map.items():
    ranked = run_map_rm3.get(str(qid), [])

    ap = ap_bioasq(ranked, relset, k=10)
    rr = rr_at_k(ranked, relset, k=10)
    s10 = success_at_k(ranked, relset, k=10)

    APs.append(ap)
    RRs.append(rr)
    S10s.append(s10)

    rec_k_vals = {}
    for K in Ks_recall:
        rK = recall_at_k(ranked, relset, k=K)
        recalls[K].append(rK)
        rec_k_vals[f"recall@{K}"] = rK

    perq.append({
        "qid": str(qid),
        "n_gold": len(relset),
        "AP@10": ap,
        "RR@10": rr,
        "Success@10": s10,
        **rec_k_vals,
    })

MAP10 = sum(APs) / len(APs) if APs else 0.0
GMAP10 = math.exp(sum(math.log(a + eps) for a in APs) / len(APs)) if APs else 0.0
MRR10 = sum(RRs) / len(RRs) if RRs else 0.0
Success10 = sum(S10s) / len(S10s) if S10s else 0.0
RecallK = {K: (sum(vals) / len(vals) if vals else 0.0) for K, vals in recalls.items()}

summary_rm3 = {
    "MRR@10": MRR10,
    "Success@10": Success10,
    "MAP@10": MAP10,
    "GMAP@10": GMAP10,
    **{f"Recall@{K}": RecallK[K] for K in Ks_recall},
    "n_queries": len(gold_map),
    "K_MAX_retrieved": K_MAX,
    "rm3_fb_docs": 10,
    "rm3_fb_terms": 10,
    "rm3_fb_lambda": 0.6,
}

print("RM3 summary:", summary_rm3)

perq_rm3_df = pl.DataFrame(perq).sort("RR@10")


# %%
if "summary" in globals():
    keys = ["MRR@10","Success@10","MAP@10","GMAP@10","Recall@50","Recall@100","Recall@200","Recall@500"]
    comp = {k: (summary.get(k), summary_rm3.get(k), (summary_rm3.get(k,0)-summary.get(k,0))) for k in keys}
    display(pl.DataFrame([{"metric": k, "bm25": v[0], "rm3": v[1], "delta": v[2]} for k,v in comp.items()]))


# %% [markdown]
# Recall@200 does not help a lot

# %% [markdown]
# # try query expansion with gene_information.tsv

# %% [markdown]
# Build a detection lookup: alias_lower -> gene_id from (Gene Name + Synonyms)
#
# - If an alias maps to multiple genes (ambiguous), we drop it from detection to avoid false triggers.
#
# Detect genes in a query:
#
# - any token matching DDB_G\d+ triggers that gene if present in the table
# - any token matching a name/synonym (case-insensitive) triggers its gene
#
# Expand the query:
#
# - For each detected gene, append a bounded list of aliases:
#   - Gene Name + Synonyms
# - Never append gene IDs (DDB_G...) even if the gene was detected via ID.
#
# Thresholds / caps (to prevent noise)
#
# - max_terms_per_gene = 6 : add at most 6 aliases per detected gene
# - max_added_total = 20 : add at most 20 aliases overall per query

# %%
# -------------------------
# 0) Load gene info
# -------------------------
GENE_INFO_PATH = os.path.abspath("../dictybase_files/gene_information.txt")

gene = pl.read_csv(GENE_INFO_PATH, separator="\t", infer_schema_length=10_000)

# normalize headers (if needed adjust these exact names)
gene = gene.rename({
    "GENE ID": "gene_id",
    "Gene Name": "gene_name",
    "Synonyms": "synonyms",
    "Gene products": "gene_products",
})

def split_syn(x):
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.upper() == "NA":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")   # ALG-2, carA-1, gpaF, etc.
DDB_RE   = re.compile(r"\bDDB_G\d+\b", re.IGNORECASE)

# -------------------------
# 1) Build lookup tables for gene detection and expansion
# -------------------------
alias_to_gene = {}    # alias_lower -> gene_id (upper), only if unambiguous
ambig_alias = set()   # alias_lower that map to multiple genes
gene_to_aliases = {}  # gene_id_upper -> ordered aliases (name+syns)
gene_to_product = {}  # gene_id_upper -> gene_product string

rows = gene.select(["gene_id", "gene_name", "synonyms", "gene_products"]).to_dicts()

for r in rows:
    gid = r.get("gene_id")
    if gid is None:
        continue
    gid = str(gid).strip()
    if not gid or gid.upper() == "NA":
        continue
    gidU = gid.upper()

    gname = r.get("gene_name")
    gname = None if gname is None else str(gname).strip()
    gname_out = gname if (gname and gname.upper() != "NA") else None

    syns = split_syn(r.get("synonyms"))

    # Get gene product
    gproduct = r.get("gene_products")
    gproduct = None if gproduct is None else str(gproduct).strip()
    gproduct_out = gproduct if (gproduct and gproduct.upper() != "NA") else None
    gene_to_product[gidU] = gproduct_out

    # expansion aliases (name + synonyms), de-dup preserving order, no gene_id
    aliases = []
    if gname_out:
        aliases.append(gname_out)
    aliases.extend(syns)

    seen = set()
    aliases = [a for a in aliases if a and (a.lower() not in seen and not seen.add(a.lower()))]
    gene_to_aliases[gidU] = aliases

    # detection mapping (unambiguous aliases only)
    for a in aliases:
        a_norm = a.lower()
        if a_norm in ambig_alias:
            continue
        if a_norm in alias_to_gene and alias_to_gene[a_norm] != gidU:
            ambig_alias.add(a_norm)
            alias_to_gene.pop(a_norm, None)
        else:
            alias_to_gene[a_norm] = gidU

print("genes:", len(gene_to_aliases))
print("detection aliases (unique):", len(alias_to_gene))
print("ambiguous aliases skipped:", len(ambig_alias))

# -------------------------
# 2) Shared gene detection function
# -------------------------
def detect_genes(query: str):
    """
    Detect genes in query by:
      - DDB_Gxxxxx tokens (collect their aliases)
      - token match against alias_to_gene (case-insensitive)
    Returns dict: {gene_id: gene_name_or_first_alias}
    """
    if query is None:
        return {}
    
    q = str(query)
    toks = TOKEN_RE.findall(q)
    toks_l = [t.lower() for t in toks]
    
    detected = {}  # gene_id -> first_mention_name
    
    # A) DDB triggers
    for m in DDB_RE.finditer(q):
        gidU = m.group(0).upper()
        if gidU in gene_to_aliases and gidU not in detected:
            detected[gidU] = gidU
    
    # B) alias triggers
    for tl in toks_l:
        gidU = alias_to_gene.get(tl)
        if gidU and gidU not in detected:
            detected[gidU] = tl.capitalize()  # Store first mention name/alias
    
    return detected


# -------------------------
# 3) Method 1: Inline expansion (original)
# -------------------------
def expand_query_inline(query: str,
                        max_terms_per_gene: int = 6,
                        max_added_total: int = 20):
    """
    Expand query by appending aliases from detected genes at the end.
    Format: "original_query alias1 alias2 alias3 ..."
    
    Returns: (expanded_query, detected_gene_ids, added_aliases)
    """
    if query is None:
        return ("", [], [])

    q = str(query)
    toks = TOKEN_RE.findall(q)
    present = set(t.lower() for t in toks)
    
    detected_map = detect_genes(q)
    detected = sorted(detected_map.keys())
    
    added = []
    for gidU in detected:
        n_gene = 0
        for a in gene_to_aliases.get(gidU, []):
            al = a.lower()
            if al in present:
                continue
            # never add DDB_G ids
            if al.startswith("ddb_g"):
                continue
            added.append(a)
            present.add(al)
            n_gene += 1
            if n_gene >= max_terms_per_gene:
                break
        if len(added) >= max_added_total:
            break

    added = added[:max_added_total]
    bm25_query = q if not added else (q + " " + " ".join(added))
    return bm25_query, detected, added


# -------------------------
# 4) Method 2: Structured expansion (new)
# -------------------------
def expand_query_structured(query: str,
                           max_aliases_per_gene: int = 4,
                           max_genes_total: int = 5,
                           include_gene_products: bool = True):
    """
    Expand query by appending structured gene information at the end.
    Format: "original_query\\ndetection(gene_name_or_alias): alias1, alias2[, product]"
    - include_gene_products=True: synonyms + gene products (long variant).
    - include_gene_products=False: id, gene name, synonyms only (synonyms variant).
    Never appends DDB_G... IDs.

    Returns: (expanded_query, detected_gene_ids, structured_info)
    """
    if query is None:
        return ("", [], "")

    q = str(query)
    toks = TOKEN_RE.findall(q)
    present = set(t.lower() for t in toks)
    
    detected_map = detect_genes(q)
    detected = sorted(detected_map.keys())
    
    # Build structured blocks
    blocks = []
    for gidU in detected[:max_genes_total]:
        first_mention = detected_map[gidU]
        aliases_list = gene_to_aliases.get(gidU, [])
        
        # Collect aliases (exclude those already in query)
        expansion_aliases = []
        for a in aliases_list:
            al = a.lower()
            if al not in present and not al.startswith("ddb_g"):
                expansion_aliases.append(a)
                if len(expansion_aliases) >= max_aliases_per_gene:
                    break
        
        # Build block: "detection(gene_name): alias1, alias2[, product]"
        expansion_str = ", ".join(expansion_aliases)
        if include_gene_products:
            product = gene_to_product.get(gidU)
            product_str = product if product else ""
            if product_str:
                if expansion_str:
                    expansion_str += ", " + product_str
                else:
                    expansion_str = product_str
        
        block = f"{first_mention}: {expansion_str}"
        blocks.append(block)
    
    # Append all blocks to query separated by " ||| "
    structured_suffix = " ||| ".join(blocks) if blocks else ""
    bm25_query = q if not structured_suffix else (q + "\n" + structured_suffix)
    
    return bm25_query, detected, structured_suffix



# %%
test_q = "rab32A is required for ..."
print("Original query:", test_q)
print("\nMethod 1 (Inline):")
print("  Expanded:", expand_query_inline(test_q)[0])
print("\nMethod 2 (Structured, synonyms only):")
print("  Expanded:", expand_query_structured(test_q, include_gene_products=False)[0])
print("\nMethod 2 (Structured, synonyms + gene products):")
print("  Expanded:", expand_query_structured(test_q, include_gene_products=True)[0])


# %%
# -------------------------
# 5) Add query_expand columns: synonyms-only and long (synonyms + gene products)
# -------------------------
gold_with_expand = gold.with_columns([
    pl.col("query").map_elements(lambda x: expand_query_structured(x, include_gene_products=False)[0], return_dtype=pl.Utf8).alias("query_expand_synonyms"),
    pl.col("query").map_elements(lambda x: expand_query_structured(x, include_gene_products=True)[0], return_dtype=pl.Utf8).alias("query_expand_long"),
])
# Backward compat: query_expand = long variant
gold_with_expand = gold_with_expand.with_columns(pl.col("query_expand_long").alias("query_expand"))

# For BM25 retrieval: use inline expansion
gold_expanded = gold.with_columns([
    pl.col("query").map_elements(lambda x: expand_query_inline(x)[0], return_dtype=pl.Utf8).alias("bm25_query"),
    pl.col("query").map_elements(lambda x: ",".join(expand_query_inline(x)[1]), return_dtype=pl.Utf8).alias("detected_genes"),
    pl.col("query").map_elements(lambda x: " ||| ".join(expand_query_inline(x)[2]), return_dtype=pl.Utf8).alias("added_aliases"),
])

with pl.Config(fmt_str_lengths=4000, tbl_rows=25, tbl_cols=20):
    print("Gold with query_expand_synonyms / query_expand_long:")
    display(gold_with_expand.select(["group_claim_id", "query", "query_expand_synonyms", "query_expand_long"]).head(2))
    print("\nGold expanded for BM25:")
    display(gold_expanded.select(["group_claim_id","query","bm25_query"]).head(2))

# %%
# -------------------------
# 6) Save gold with query_expand
# -------------------------
# Save as parquet
gold_with_expand.write_parquet("../output/cleaned/gold_with_query_expand.parquet")
print("Saved: ../output/cleaned/gold_with_query_expand.parquet")

# Flatten (unnest docs) and save as TSV
gold_flat = (
    gold_with_expand
    .select(["group_claim_id", "query", "query_expand_synonyms", "query_expand_long", "query_expand", "docs"])
    .explode("docs")
    .with_columns([
        pl.col("docs").struct.field("pmid").alias("pmid"),
        pl.col("docs").struct.field("title").alias("title"),
        pl.col("docs").struct.field("abstract_clean").alias("abstract_clean"),
        pl.col("docs").struct.field("year").alias("year"),
    ])
    .drop("docs")
)

gold_flat.write_csv("../output/cleaned/gold_with_query_expand_flat.tsv", separator="\t")
print("Saved: ../output/cleaned/gold_with_query_expand_flat.tsv")
print(f"Rows in flat file: {len(gold_flat)}")


# %%
# -------------------------
# 7) Run BM25 with Inline Expansion
# -------------------------
queries_df_expanded = gold_expanded.select([
    pl.col("group_claim_id").cast(pl.Utf8).alias("qid"),
    pl.col("bm25_query").alias("query"),
])

qdf_expanded = queries_df_expanded.to_pandas()

K_MAX = 500
res_all_expanded = br.transform(qdf_expanded)

res_expanded = (
    res_all_expanded
    .sort_values(["qid", "rank"], ascending=[True, True])
    .groupby("qid", as_index=False)
    .head(K_MAX)
)
res_expanded["qid"] = res_expanded["qid"].astype(str)
res_expanded["docno"] = res_expanded["docno"].astype(str)
res_expanded = res_expanded.sort_values(["qid", "rank"], ascending=[True, True])

# Build run_map for expanded queries
run_map_expanded = {}
for qid, grp in res_expanded.groupby("qid", sort=False):
    run_map_expanded[qid] = grp["docno"].tolist()

print("Expanded queries retrieved:", len(run_map_expanded))

# %%
# -------------------------
# 8) Evaluate expanded queries
# -------------------------
perq_expanded = []
APs_exp, RRs_exp, S10s_exp = [], [], []
recalls_exp = {K: [] for K in Ks_recall}

for qid, relset in gold_map.items():
    ranked = run_map_expanded.get(qid, [])

    ap = ap_bioasq(ranked, relset, k=10)
    rr = rr_at_k(ranked, relset, k=10)
    s10 = success_at_k(ranked, relset, k=10)

    APs_exp.append(ap)
    RRs_exp.append(rr)
    S10s_exp.append(s10)

    rec_k_vals = {}
    for K in Ks_recall:
        rK = recall_at_k(ranked, relset, k=K)
        recalls_exp[K].append(rK)
        rec_k_vals[f"recall@{K}"] = rK

    perq_expanded.append({
        "qid": qid,
        "n_gold": len(relset),
        "AP@10": ap,
        "RR@10": rr,
        "Success@10": s10,
        **rec_k_vals,
    })

MAP10_exp = sum(APs_exp) / len(APs_exp) if APs_exp else 0.0
GMAP10_exp = math.exp(sum(math.log(a + eps) for a in APs_exp) / len(APs_exp)) if APs_exp else 0.0
MRR10_exp = sum(RRs_exp) / len(RRs_exp) if RRs_exp else 0.0
Success10_exp = sum(S10s_exp) / len(S10s_exp) if S10s_exp else 0.0
RecallK_exp = {K: (sum(vals) / len(vals) if vals else 0.0) for K, vals in recalls_exp.items()}

summary_expanded = {
    "MRR@10": MRR10_exp,
    "Success@10": Success10_exp,
    "MAP@10": MAP10_exp,
    "GMAP@10": GMAP10_exp,
    **{f"Recall@{K}": RecallK_exp[K] for K in Ks_recall},
    "n_queries": len(gold_map),
    "K_MAX_retrieved": K_MAX,
}

print("EXPANDED query summary:")
print(summary_expanded)

perq_expanded_df = pl.DataFrame(perq_expanded).sort("RR@10")

# %%
# -------------------------
# 9) Compare: Baseline vs Inline Expansion
# -------------------------
metrics = ["MRR@10", "Success@10", "MAP@10", "GMAP@10", "Recall@50", "Recall@100", "Recall@200", "Recall@500"]

comparison = []
for m in metrics:
    baseline = summary.get(m, 0.0)
    expanded = summary_expanded.get(m, 0.0)
    
    delta = expanded - baseline
    delta_pct = (delta / baseline * 100) if baseline != 0 else 0.0
    
    comparison.append({
        "metric": m,
        "baseline": baseline,
        "expanded_inline": expanded,
        "delta": delta,
        "delta_%": delta_pct,
    })

comparison_df = pl.DataFrame(comparison)
print(comparison_df)


# %% [markdown]
# this seems like a solid win

# %% [markdown]
# # RM3 on expanded queries

# %%
# Run RM3 on expanded queries
rm3_pipe_expanded = br >> rm3 >> br
res_all_rm3_expanded = rm3_pipe_expanded.transform(qdf_expanded)

res_rm3_expanded = (
    res_all_rm3_expanded
    .sort_values(["qid", "rank"], ascending=[True, True])
    .groupby("qid", as_index=False)
    .head(K_MAX)
)
res_rm3_expanded["qid"] = res_rm3_expanded["qid"].astype(str)
res_rm3_expanded["docno"] = res_rm3_expanded["docno"].astype(str)
res_rm3_expanded = res_rm3_expanded.sort_values(["qid", "rank"], ascending=[True, True])

# build run map
run_map_rm3_expanded = {}
for qid, grp in res_rm3_expanded.groupby("qid", sort=False):
    run_map_rm3_expanded[qid] = grp["docno"].tolist()

print("RM3 on expanded queries - retrieved:", len(run_map_rm3_expanded))


# %%
# Evaluate RM3 on expanded
perq_rm3_expanded = []
APs_rm3_exp, RRs_rm3_exp, S10s_rm3_exp = [], [], []
recalls_rm3_exp = {K: [] for K in Ks_recall}

for qid, relset in gold_map.items():
    ranked = run_map_rm3_expanded.get(str(qid), [])

    ap = ap_bioasq(ranked, relset, k=10)
    rr = rr_at_k(ranked, relset, k=10)
    s10 = success_at_k(ranked, relset, k=10)

    APs_rm3_exp.append(ap)
    RRs_rm3_exp.append(rr)
    S10s_rm3_exp.append(s10)

    rec_k_vals = {}
    for K in Ks_recall:
        rK = recall_at_k(ranked, relset, k=K)
        recalls_rm3_exp[K].append(rK)
        rec_k_vals[f"recall@{K}"] = rK

    perq_rm3_expanded.append({
        "qid": str(qid),
        "n_gold": len(relset),
        "AP@10": ap,
        "RR@10": rr,
        "Success@10": s10,
        **rec_k_vals,
    })

MAP10_rm3_exp = sum(APs_rm3_exp) / len(APs_rm3_exp) if APs_rm3_exp else 0.0
GMAP10_rm3_exp = math.exp(sum(math.log(a + eps) for a in APs_rm3_exp) / len(APs_rm3_exp)) if APs_rm3_exp else 0.0
MRR10_rm3_exp = sum(RRs_rm3_exp) / len(RRs_rm3_exp) if RRs_rm3_exp else 0.0
Success10_rm3_exp = sum(S10s_rm3_exp) / len(S10s_rm3_exp) if S10s_rm3_exp else 0.0
RecallK_rm3_exp = {K: (sum(vals) / len(vals) if vals else 0.0) for K, vals in recalls_rm3_exp.items()}

summary_rm3_expanded = {
    "MRR@10": MRR10_rm3_exp,
    "Success@10": Success10_rm3_exp,
    "MAP@10": MAP10_rm3_exp,
    "GMAP@10": GMAP10_rm3_exp,
    **{f"Recall@{K}": RecallK_rm3_exp[K] for K in Ks_recall},
    "n_queries": len(gold_map),
    "K_MAX_retrieved": K_MAX,
}

print("RM3 on EXPANDED query summary:")
print(summary_rm3_expanded)

perq_rm3_expanded_df = pl.DataFrame(perq_rm3_expanded).sort("RR@10")


# %%
# Compare all 4 approaches
metrics = ["MRR@10", "Success@10", "MAP@10", "GMAP@10", "Recall@50", "Recall@100", "Recall@200", "Recall@500"]

all_comp = []
for m in metrics:
    bm25_base = summary.get(m, 0.0)
    bm25_exp = summary_expanded.get(m, 0.0)
    rm3_base = summary_rm3.get(m, 0.0)
    rm3_exp = summary_rm3_expanded.get(m, 0.0)
    
    all_comp.append({
        "metric": m,
        "BM25": bm25_base,
        "BM25+Expanded": bm25_exp,
        "BM25+RM3": rm3_base,
        "BM25+Expanded+RM3": rm3_exp,
    })

comp_all_df = pl.DataFrame(all_comp)
with pl.Config(fmt_str_lengths=200, tbl_rows=20, tbl_cols=20):
    display(comp_all_df)

print("\n=== SUMMARY ===")
print("\n1. BM25 (original query):")
print({k: v for k, v in summary.items() if k not in ["n_queries", "K_MAX_retrieved"]})
print("\n2. BM25 + Expanded queries:")
print({k: v for k, v in summary_expanded.items() if k not in ["n_queries", "K_MAX_retrieved"]})
print("\n3. BM25 + RM3 (original):")
print({k: v for k, v in summary_rm3.items() if k not in ["n_queries", "K_MAX_retrieved", "rm3_fb_docs", "rm3_fb_terms", "rm3_fb_lambda"]})
print("\n4. BM25 + Expanded + RM3:")
print({k: v for k, v in summary_rm3_expanded.items() if k not in ["n_queries", "K_MAX_retrieved"]})


# %% [markdown]
# we don't do RM3

# %%

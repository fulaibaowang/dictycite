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
# # Qualitative examples for the paper
#
# Goal: surface a small number of concrete query-level examples to illustrate each of the three
# main findings (rerankers, gene-aware query expansion, full-text retrieval).
#
# Design principle: each example is an **existence proof of a mechanism**, not evidence for the
# population. Population evidence lives in Table 1 / Fig 4 / Fig 5 / Table S1 / Fig S2 / Fig S4.
# We pick examples that make the mechanism legible in a small table cell; selection is
# transparent (filter rules are in the cells below).
#
# **Outputs:** the final cells produce short markdown snippets that can be pasted into
# `output/paper_figures/paper.tex`. Body-chunk text lookup for Ex.3 is provided as a helper.

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl

REPO = Path("/Users/yun/develop/dictycite")
OUT = REPO / "output"

# %% [markdown]
# ## 1. Paths
#
# All three findings can be reproduced from the public goldset (`7a`) and the
# query-expansion benchmark (`7d`). For each finding, we point to the workflow whose
# `rerank/post_rerank_fusion/runs/*.tsv` gives the final ranking under that system.

# %%
GOLDSET_7A = OUT / "dicty_gold_build" / "7a_dicty_gold_llm_public.jsonl"
QE_BENCH_7D = OUT / "dicty_gold_build" / "7d_dicty_gold_query_expansion_benchmark.jsonl"
ABSTRACTS_PARQUET = OUT / "dicty_gold_build" / "3_articles_cleaned_abstract.parquet"
CHUNKS_JSONL = OUT / "pdf_extraction" / "v2" / "chunks.jsonl"

# Ex.1 — reranker comparison (abstract corpus, full 1,656-query goldset)
RUN_BM25_7A = (
    OUT
    / "workflow_frida_7a_public_goldset_rerank_gemma"
    / "retrieval/bm25/runs/BM25__7a_dicty_gold_llm_public__top5000.tsv"
)
RUN_MSMARCO_7A = (
    OUT
    / "workflow_frida_7a_public_goldset_rerank_ms_marco_minilm"
    / "rerank/post_rerank_fusion/runs"
    / "best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR50_poolH50_k60.tsv"
)
RUN_BGEM3_7A = (
    OUT
    / "workflow_frida_7a_public_goldset_both_routes"
    / "rerank/post_rerank_fusion/runs"
    / "best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR50_poolH50_k60.tsv"
)
RUN_MEDCPT_7A = (
    OUT
    / "workflow_vega_7a_public_goldset_rerank_medcpt"
    / "rerank/post_rerank_fusion/runs"
    / "best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR50_poolH50_k60.tsv"
)
RUN_GEMMA_7A = (
    OUT
    / "workflow_frida_7a_public_goldset_rerank_gemma"
    / "rerank/post_rerank_fusion/runs"
    / "best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR50_poolH50_k60.tsv"
)

# Ex.2 — QE rerank-query sweep (fixed deepest-expansion candidate pool, vary reranker query)
# Source dataset is 7d (563-query expansion benchmark). BGE-m3 sweep (matches Table S1 row).
RUN_QE_BGEM3_BODY = (
    OUT
    / "workflow_baseline_full_sweep/workflow_fixed_long_rerank_sweep_7d"
    / "fixed_long_rerank_sweep/rerank_body/runs"
    / "best_rrf_7d_dicty_gold_query_expansion_benchmark_top5000.tsv"
)
RUN_QE_BGEM3_SYN = (
    OUT
    / "workflow_baseline_full_sweep/workflow_fixed_long_rerank_sweep_7d"
    / "fixed_long_rerank_sweep/rerank_synonyms/runs"
    / "best_rrf_7d_dicty_gold_query_expansion_benchmark_top5000.tsv"
)
RUN_QE_BGEM3_LONG = (
    OUT
    / "workflow_baseline_full_sweep/workflow_fixed_long_rerank_sweep_7d"
    / "fixed_long_rerank_sweep/rerank_long/runs"
    / "best_rrf_7d_dicty_gold_query_expansion_benchmark_top5000.tsv"
)

# Ex.3 — abstract-only vs chunked full-text corpus (BGE-m3, 7a goldset)
RUN_ABS_BGEM3_7A = RUN_BGEM3_7A  # same as Ex.1 BGE-m3
RUN_CHUNK_BGEM3_7A = (
    OUT
    / "workflow_frida_7a_public_goldset_chunked_v2"
    / "rerank/post_rerank_fusion/runs"
    / "best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR300_poolH300_k60.tsv"
)

# Sanity-check every path before we go further.
for p in [
    GOLDSET_7A, QE_BENCH_7D, ABSTRACTS_PARQUET, CHUNKS_JSONL,
    RUN_BM25_7A, RUN_MSMARCO_7A, RUN_BGEM3_7A, RUN_MEDCPT_7A, RUN_GEMMA_7A,
    RUN_QE_BGEM3_BODY, RUN_QE_BGEM3_SYN, RUN_QE_BGEM3_LONG,
    RUN_ABS_BGEM3_7A, RUN_CHUNK_BGEM3_7A,
]:
    assert p.exists(), f"missing: {p}"
print("all paths ok")

# %% [markdown]
# ## 2. Loaders

# %%
def load_goldset(path: Path) -> pl.DataFrame:
    """One row per query: qid, query_text, list of gold pmids, evidence_level (per doc), abstracts."""
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({
                "qid": int(r["query_id"]),
                "query_text": r["query_text"],
                "gold_pmids": [str(d["pmid"]) for d in r["docs"]],
                "gold_titles": [d["title"] for d in r["docs"]],
                "gold_abstracts": [d["abstract_clean"] for d in r["docs"]],
                "evidence_levels": [d["evidence_level"] for d in r["docs"]],
                "genes": r.get("genes", []),
            })
    return pl.DataFrame(rows)


def load_qe_benchmark(path: Path) -> pl.DataFrame:
    """One row per QE-eligible query: qid, original, +syn, +syn&prod, genes, gold pmids/titles."""
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({
                "qid": int(r["query_id"]),
                "query_text_body": r["query_text"],
                "query_text_synonyms": r.get("query_text_expansion_synonyms"),
                "query_text_long": r.get("query_text_synonym_products"),
                "gold_pmids": [str(d["pmid"]) for d in r["docs"]],
                "gold_titles": [d["title"] for d in r["docs"]],
                "gold_abstracts": [d["abstract_clean"] for d in r["docs"]],
                "detected_genes": r.get("query_gene_expansion", {}).get("detected_genes", []),
            })
    return pl.DataFrame(rows)


def load_run(path: Path) -> pl.DataFrame:
    """Run TSV: qid, docno, rank (and optionally score). Returns standardized cols,
    normalized to 1-indexed ranks (BM25 files are 0-indexed; rerank files are 1-indexed)."""
    df = pl.read_csv(path, separator="\t", schema_overrides={"qid": pl.Int64, "docno": pl.Utf8})
    df = df.select(["qid", "docno", "rank"])
    if df["rank"].min() == 0:
        df = df.with_columns((pl.col("rank") + 1).alias("rank"))
    return df


def rank_of_gold(run: pl.DataFrame, gold_pmids: list[str]) -> int | None:
    """Best (lowest) rank among gold pmids for this run-restricted-to-one-qid. None if missing."""
    matched = run.filter(pl.col("docno").is_in(gold_pmids))
    if matched.is_empty():
        return None
    return int(matched["rank"].min())


# %%
gold = load_goldset(GOLDSET_7A)
qe_bench = load_qe_benchmark(QE_BENCH_7D)
print(f"goldset: {len(gold)} queries; QE benchmark: {len(qe_bench)} queries")
gold.head(2)

# %% [markdown]
# ## 3. Wide per-query frame for Ex.1 (reranker comparison)
#
# Compute rank-of-gold under each of BM25, MS-MARCO, BGE-m3, MedCPT, BGE-Gemma.
# Then for each query also surface the **top-ranked non-gold article** under MS-MARCO
# (the "competitor" we'll show in the table).

# %%
runs_ex1 = {
    "BM25": load_run(RUN_BM25_7A),
    "MS-MARCO": load_run(RUN_MSMARCO_7A),
    "BGE-m3": load_run(RUN_BGEM3_7A),
    "MedCPT": load_run(RUN_MEDCPT_7A),
    "Gemma": load_run(RUN_GEMMA_7A),
}

# Group each run by qid so we can do per-query lookups efficiently.
runs_ex1_by_qid = {
    name: {key[0]: g.drop("qid") for key, g in df.group_by("qid")}
    for name, df in runs_ex1.items()
}


def rank_of_gold_for_qid(name: str, qid: int, gold_pmids: list[str]) -> int | None:
    g = runs_ex1_by_qid[name].get(qid)
    if g is None:
        return None
    return rank_of_gold(g, gold_pmids)


def top_nongold(name: str, qid: int, gold_pmids: list[str], k: int = 1) -> list[str]:
    g = runs_ex1_by_qid[name].get(qid)
    if g is None:
        return []
    nongold = g.filter(~pl.col("docno").is_in(gold_pmids)).sort("rank").head(k)
    return nongold["docno"].to_list()


# Build wide frame.
rows = []
for r in gold.iter_rows(named=True):
    qid = r["qid"]
    gpmids = r["gold_pmids"]
    rec = {
        "qid": qid,
        "query_text": r["query_text"],
        "gold_pmids": gpmids,
        "gold_titles": r["gold_titles"],
    }
    for name in runs_ex1:
        rec[f"rank_{name}"] = rank_of_gold_for_qid(name, qid, gpmids)
    rec["top1_nongold_msmarco"] = (top_nongold("MS-MARCO", qid, gpmids, 1) or [None])[0]
    rec["top1_nongold_bm25"] = (top_nongold("BM25", qid, gpmids, 1) or [None])[0]
    rows.append(rec)

wide_ex1 = pl.DataFrame(rows)
print(wide_ex1.shape)
wide_ex1.head(3)

# %% [markdown]
# ## 4. Ex.1 candidates
#
# Two examples to illustrate the discussion sentence:
# > A reranker helps only when it preserves the gene/phenotype signal and adds useful
# > discrimination; a weaker/general reranker can hurt.
#
# We surface a handful of strong candidates for each direction. Selection is by inspection —
# we look at the rank columns + query text + the "competitor" (top non-gold article under
# the relevant reranker) and pick one whose mechanism is legible in a small table cell.

# %% [markdown]
# ### Ex.1a — "Lexical signal lost"
#
# Cases where BM25 ranks the gold high but MS-MARCO MiniLM demotes it severely.
# Strong cases: `rank_BM25 ≤ 3` and `rank_MS-MARCO ≥ 50` and `rank_Gemma ≤ 3`
# (so Gemma confirms the gold really is the right answer; the MS-MARCO drop isn't
# just because the gold is borderline).

# %%
ex1a_pool = (
    wide_ex1
    .filter(
        pl.col("rank_BM25").is_not_null()
        & pl.col("rank_MS-MARCO").is_not_null()
        & pl.col("rank_Gemma").is_not_null()
    )
    # Moderate filter: BM25 ranks gold near top, MS-MARCO meaningfully demotes it,
    # Gemma still ranks it high (confirming gold is genuinely the right answer).
    .filter(pl.col("rank_BM25") <= 5)
    .filter(pl.col("rank_MS-MARCO") >= 20)
    .filter(pl.col("rank_Gemma") <= 5)
    .with_columns(
        (pl.col("rank_MS-MARCO") - pl.col("rank_BM25")).alias("msmarco_drop")
    )
    .sort("msmarco_drop", descending=True)
)
print(f"Ex.1a pool size: {len(ex1a_pool)}")
ex1a_pool.select(
    ["qid", "rank_BM25", "rank_MS-MARCO", "rank_BGE-m3", "rank_MedCPT", "rank_Gemma",
     "query_text", "top1_nongold_msmarco"]
).head(8)

# %% [markdown]
# ### Ex.1b — "Semantic rescue"
#
# Cases where BM25 misses the gold (vocabulary gap) but a strong reranker bridges it.
# Strong cases: `rank_BM25 ≥ 20` and `rank_Gemma ≤ 3`. We additionally require the
# gold's title to share **few** content words with the query (proxy for vocabulary gap).

# %%
import re

_STOP = {
    "a","an","the","and","or","but","of","in","on","to","for","with","by","is","are",
    "was","were","be","been","being","that","this","these","those","as","at","from",
    "it","its","their","his","her","he","she","they","we","you","i","not","no","so",
    "if","then","than","also","into","via","such","using","via","can","may",
    "dictyostelium","discoideum",
}

def content_tokens(s: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", s.lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}

def lex_overlap_query_vs_titles(query: str, titles: list[str]) -> float:
    q = content_tokens(query)
    if not q:
        return 0.0
    best = 0.0
    for t in titles:
        tt = content_tokens(t)
        if not tt:
            continue
        ov = len(q & tt) / len(q)
        best = max(best, ov)
    return best


wide_ex1b = wide_ex1.with_columns(
    pl.struct(["query_text", "gold_titles"])
    .map_elements(lambda s: lex_overlap_query_vs_titles(s["query_text"], s["gold_titles"]),
                  return_dtype=pl.Float64)
    .alias("query_title_overlap")
)

ex1b_pool = (
    wide_ex1b
    .filter(
        pl.col("rank_BM25").is_not_null()
        & pl.col("rank_Gemma").is_not_null()
    )
    # Moderate filter: BM25 misses the gold (vocabulary gap), Gemma recovers it.
    .filter(pl.col("rank_BM25") >= 10)
    .filter(pl.col("rank_Gemma") <= 5)
    .filter(pl.col("query_title_overlap") <= 0.25)
    .with_columns(
        (pl.col("rank_BM25") - pl.col("rank_Gemma")).alias("gemma_lift")
    )
    .sort("gemma_lift", descending=True)
)
print(f"Ex.1b pool size: {len(ex1b_pool)}")
ex1b_pool.select(
    ["qid", "rank_BM25", "rank_MS-MARCO", "rank_BGE-m3", "rank_MedCPT", "rank_Gemma",
     "query_title_overlap", "query_text", "gold_titles", "top1_nongold_bm25"]
).head(8)

# %% [markdown]
# ### Drill-down helper: show the query, the gold abstract, and the competitor abstract.

# %%
abstracts = pl.read_parquet(ABSTRACTS_PARQUET)
# Inspect columns to find the right names for pmid + title + abstract.
print(abstracts.columns)
abstracts.head(2)

# %%
def lookup_article(pmid: str) -> dict:
    rec = abstracts.filter(pl.col("pmid").cast(pl.Utf8) == str(pmid))
    if rec.is_empty():
        return {"pmid": pmid, "title": None, "abstract": None}
    r = rec.row(0, named=True)
    title_col = "title" if "title" in r else next((c for c in r if "title" in c.lower()), None)
    abs_col = (
        "abstract_clean" if "abstract_clean" in r
        else next((c for c in r if "abstract" in c.lower()), None)
    )
    return {"pmid": pmid, "title": r.get(title_col), "abstract": r.get(abs_col)}


def drill_ex1(qid: int) -> None:
    row = wide_ex1.filter(pl.col("qid") == qid).row(0, named=True)
    print("=" * 80)
    print(f"qid {qid}  |  query: {row['query_text']}")
    print(
        f"  rank_BM25={row['rank_BM25']}  rank_MS-MARCO={row['rank_MS-MARCO']}  "
        f"rank_BGE-m3={row['rank_BGE-m3']}  rank_MedCPT={row['rank_MedCPT']}  "
        f"rank_Gemma={row['rank_Gemma']}"
    )
    for i, (pmid, title) in enumerate(zip(row["gold_pmids"], row["gold_titles"])):
        print(f"  GOLD #{i+1}  PMID {pmid}  |  {title}")
    for label, key in [("competitor (MS-MARCO top-1)", "top1_nongold_msmarco"),
                       ("competitor (BM25 top-1)", "top1_nongold_bm25")]:
        pmid = row.get(key)
        if pmid:
            art = lookup_article(pmid)
            print(f"  {label}  PMID {pmid}  |  {art['title']}")


# Example: drill into a few of the top candidates from each pool.
for qid in ex1a_pool.head(3)["qid"].to_list():
    drill_ex1(qid)
for qid in ex1b_pool.head(3)["qid"].to_list():
    drill_ex1(qid)

# %% [markdown]
# ## 5. Ex.2 candidates — gene-aware query expansion
#
# Build a small wide frame restricted to the QE benchmark (n=563): rank-of-gold under
# BGE-m3 reranker when the query passed to the reranker is `body` / `+synonyms` / `+long`,
# with the candidate pool fixed (this is the experimental setup behind Table S1).
#
# We look for queries where +synonyms rescues the gold and where the added synonym appears
# verbatim in the gold title or abstract — that makes the mechanism legible.

# %%
runs_ex2 = {
    "body": load_run(RUN_QE_BGEM3_BODY),
    "+syn": load_run(RUN_QE_BGEM3_SYN),
    "+long": load_run(RUN_QE_BGEM3_LONG),
}
runs_ex2_by_qid = {
    name: {key[0]: g.drop("qid") for key, g in df.group_by("qid")}
    for name, df in runs_ex2.items()
}

rows = []
for r in qe_bench.iter_rows(named=True):
    qid = r["qid"]
    gpmids = r["gold_pmids"]
    rec = {
        "qid": qid,
        "query_body": r["query_text_body"],
        "query_syn": r["query_text_synonyms"],
        "query_long": r["query_text_long"],
        "gold_pmids": gpmids,
        "gold_titles": r["gold_titles"],
        "gold_abstracts": r["gold_abstracts"],
        "detected_genes": r["detected_genes"],
    }
    for name in runs_ex2:
        g = runs_ex2_by_qid[name].get(qid)
        rec[f"rank_{name}"] = rank_of_gold(g, gpmids) if g is not None else None
    rows.append(rec)

wide_ex2 = pl.DataFrame(rows)
print(wide_ex2.shape)
wide_ex2.head(3)

# %%
def added_terms(orig: str, expanded: str | None) -> str:
    """Crude diff: tokens in `expanded` that are not in `orig`, preserving order."""
    if not expanded:
        return ""
    orig_toks = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]+", orig.lower()))
    out, seen = [], set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", expanded):
        if tok.lower() in orig_toks:
            continue
        if tok.lower() in seen:
            continue
        seen.add(tok.lower())
        out.append(tok)
    return " ".join(out)


def added_term_in_gold(added: str, titles: list[str], abstracts: list[str]) -> bool:
    if not added:
        return False
    text = " ".join(titles + abstracts).lower()
    for tok in added.split():
        if tok.lower() in text:
            return True
    return False


wide_ex2 = wide_ex2.with_columns([
    pl.struct(["query_body", "query_syn"])
    .map_elements(lambda s: added_terms(s["query_body"], s["query_syn"]),
                  return_dtype=pl.Utf8)
    .alias("added_syn"),
    pl.struct(["query_body", "query_long"])
    .map_elements(lambda s: added_terms(s["query_body"], s["query_long"]),
                  return_dtype=pl.Utf8)
    .alias("added_long"),
])
wide_ex2 = wide_ex2.with_columns(
    pl.struct(["added_syn", "gold_titles", "gold_abstracts"])
    .map_elements(lambda s: added_term_in_gold(s["added_syn"], s["gold_titles"], s["gold_abstracts"]),
                  return_dtype=pl.Boolean)
    .alias("syn_in_gold")
)

ex2_pool = (
    wide_ex2
    .filter(pl.col("rank_body").is_not_null() & pl.col("rank_+syn").is_not_null())
    .filter(pl.col("rank_body") >= 10)
    .filter(pl.col("rank_+syn") <= 3)
    .filter(pl.col("syn_in_gold"))
    .with_columns(
        (pl.col("rank_body") - pl.col("rank_+syn")).alias("syn_lift")
    )
    .sort("syn_lift", descending=True)
)
print(f"Ex.2 pool size: {len(ex2_pool)}")
ex2_pool.select(
    ["qid", "rank_body", "rank_+syn", "rank_+long",
     "added_syn", "query_body", "gold_titles"]
).head(8)

# %% [markdown]
# ## 6. Ex.3 candidates — full-text chunks help abstract-insufficient claims
#
# Restrict to queries whose gold doc is labeled `abstract_insufficient`, then find ones
# where the rank improves from "abstract-only BGE-m3" to "+chunks BGE-m3".
#
# For each candidate we also need a body-chunk excerpt that contains the cue missing from
# the abstract. The helper below pulls the top-ranked chunks for the gold PMID from the
# chunked corpus.

# %%
runs_ex3 = {
    "abs": load_run(RUN_ABS_BGEM3_7A),
    "chunks": load_run(RUN_CHUNK_BGEM3_7A),
}
runs_ex3_by_qid = {
    name: {key[0]: g.drop("qid") for key, g in df.group_by("qid")}
    for name, df in runs_ex3.items()
}


def insufficient_gold_pmids(r: dict) -> list[str]:
    return [
        str(p)
        for p, lvl in zip(r["gold_pmids"], r["evidence_levels"])
        if lvl == "abstract_insufficient"
    ]


rows = []
for r in gold.iter_rows(named=True):
    insuff = insufficient_gold_pmids(r)
    if not insuff:
        continue
    qid = r["qid"]
    rec = {
        "qid": qid,
        "query_text": r["query_text"],
        "gold_pmids": r["gold_pmids"],
        "gold_titles": r["gold_titles"],
        "gold_abstracts": r["gold_abstracts"],
        "insufficient_pmids": insuff,
    }
    g_abs = runs_ex3_by_qid["abs"].get(qid)
    g_chk = runs_ex3_by_qid["chunks"].get(qid)
    # For the chunked corpus, doc IDs include chunk suffixes like "PMID#body_001";
    # collapse to PMID first-occurrence.
    if g_chk is not None:
        g_chk = (
            g_chk.with_columns(pl.col("docno").str.split("#").list.first().alias("pmid"))
            .sort("rank")
            .group_by("pmid", maintain_order=True)
            .agg(pl.col("rank").min().alias("rank"))
            .rename({"pmid": "docno"})
        )
    rec["rank_abs_any_gold"] = rank_of_gold(g_abs, r["gold_pmids"]) if g_abs is not None else None
    rec["rank_chk_any_gold"] = rank_of_gold(g_chk, r["gold_pmids"]) if g_chk is not None else None
    rec["rank_abs_insuff"] = rank_of_gold(g_abs, insuff) if g_abs is not None else None
    rec["rank_chk_insuff"] = rank_of_gold(g_chk, insuff) if g_chk is not None else None
    rows.append(rec)

wide_ex3 = pl.DataFrame(rows)
print(wide_ex3.shape)

ex3_pool = (
    wide_ex3
    .filter(
        pl.col("rank_abs_insuff").is_not_null()
        & pl.col("rank_chk_insuff").is_not_null()
    )
    .filter(pl.col("rank_abs_insuff") >= 50)
    .filter(pl.col("rank_chk_insuff") <= 10)
    # Single insufficient gold PMID makes the example unambiguous.
    .filter(pl.col("insufficient_pmids").list.len() == 1)
    .with_columns(
        (pl.col("rank_abs_insuff") - pl.col("rank_chk_insuff")).alias("chunk_lift")
    )
    .sort("chunk_lift", descending=True)
)
print(f"Ex.3 pool size: {len(ex3_pool)}")
ex3_pool.select(
    ["qid", "rank_abs_insuff", "rank_chk_insuff", "query_text", "gold_titles",
     "insufficient_pmids"]
).head(8)

# %% [markdown]
# ### Body-chunk lookup
#
# For a given (qid, gold pmid), surface the top-ranked body chunks for that PMID under the
# chunked-corpus run, then pull their text from `pdf_extraction/v2/chunks.jsonl`.

# %%
# Build a chunk-text index once. Indexed by chunk_id.
chunk_text: dict[str, dict] = {}
with open(CHUNKS_JSONL) as fh:
    for line in fh:
        c = json.loads(line)
        chunk_text[c["chunk_id"]] = {
            "pmid": str(c["pmid"]),
            "type": c.get("type"),
            "seq": c.get("seq"),
            "text": c["text"],
        }
print(f"chunks indexed: {len(chunk_text)}")

# Re-load the chunked rerank run preserving the chunk-level docnos so we can find which
# chunks scored highest.
chunked_run_full = load_run(RUN_CHUNK_BGEM3_7A)
chunked_run_full = chunked_run_full.with_columns(
    pl.col("docno").str.split("#").list.first().alias("pmid")
)


def top_chunks_for_pmid(qid: int, pmid: str, k: int = 3) -> list[dict]:
    rows = chunked_run_full.filter(
        (pl.col("qid") == qid) & (pl.col("pmid") == str(pmid))
    ).sort("rank").head(k)
    out = []
    for r in rows.iter_rows(named=True):
        info = chunk_text.get(r["docno"], {})
        out.append({"rank": r["rank"], "chunk_id": r["docno"], **info})
    return out


def drill_ex3(qid: int) -> None:
    row = wide_ex3.filter(pl.col("qid") == qid).row(0, named=True)
    print("=" * 80)
    print(f"qid {qid}  |  query: {row['query_text']}")
    print(
        f"  rank_abs (insuff)={row['rank_abs_insuff']}   "
        f"rank_chunks (insuff)={row['rank_chk_insuff']}"
    )
    for pmid in row["insufficient_pmids"]:
        idx = row["gold_pmids"].index(pmid)
        title = row["gold_titles"][idx]
        abstract = row["gold_abstracts"][idx]
        print(f"  GOLD PMID {pmid}  |  {title}")
        print(f"  ABSTRACT: {abstract[:400]}...")
        for ch in top_chunks_for_pmid(qid, pmid, k=3):
            txt = (ch.get("text") or "")[:400].replace("\n", " ")
            print(f"  chunk rank={ch['rank']}  {ch['chunk_id']}  ({ch.get('type')}): {txt}...")


for qid in ex3_pool.head(3)["qid"].to_list():
    drill_ex3(qid)

# %% [markdown]
# ## 7. Sentence-excerpt helpers
#
# To make each example legible in a compact table, we surface **one short sentence**
# from the abstract/body — the one whose content tokens best overlap with the query.
# Sentences are trimmed to ~180 characters with an ellipsis.

# %%
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Light cleanup; strip section-header artifacts and excessive whitespace.
    t = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]


def best_sentence(
    query: str, text: str, must_include: list[str] | None = None, max_chars: int = 180
) -> str:
    """Pick the sentence whose content-token overlap with `query` is highest.

    If `must_include` is given, restrict to sentences containing any of those tokens
    (case-insensitive substring), then pick the best by query overlap.
    Returns truncated string with ellipsis.
    """
    sents = split_sentences(text)
    if not sents:
        return ""
    qtoks = content_tokens(query)
    if must_include:
        needles = [n.lower() for n in must_include if n]
        filtered = [
            s for s in sents
            if any(n in s.lower() for n in needles)
        ]
        if filtered:
            sents = filtered
    scored = []
    for s in sents:
        st = content_tokens(s)
        ov = len(qtoks & st) / max(len(qtoks), 1)
        scored.append((ov, len(s), s))
    # Highest overlap; tie-break by shorter sentence (more focused).
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0][2]
    return shorten(best, max_chars)


def shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def short_title(title: str, max_chars: int = 75) -> str:
    return shorten(title or "", max_chars)


# %% [markdown]
# ### Unified candidate inspector
#
# Prints the top-N candidates from each pool with full mechanism context (query,
# gold title + best sentence, competitor title + best sentence, ranks). Use this
# to pick the SELECTED_* qids.

# %%
def show_ex1_candidates(pool: pl.DataFrame, label: str, n: int = 5) -> None:
    print(f"\n{'=' * 80}\nEx.1 — {label}  (top {n} of {len(pool)})\n{'=' * 80}")
    for r in pool.head(n).iter_rows(named=True):
        qid = r["qid"]
        gold_pmid = r["gold_pmids"][0]
        gold_title = r["gold_titles"][0]
        gold_abs = lookup_article(gold_pmid)["abstract"] or ""
        gold_sent = best_sentence(r["query_text"], gold_abs)
        comp_key = "top1_nongold_msmarco" if "lost" in label.lower() else "top1_nongold_bm25"
        comp_pmid = r[comp_key]
        comp = lookup_article(comp_pmid) if comp_pmid else None
        comp_sent = best_sentence(r["query_text"], comp["abstract"] or "") if comp else ""
        print(f"\nqid={qid}  BM25={r['rank_BM25']}  MS-MARCO={r['rank_MS-MARCO']}  "
              f"BGE-m3={r['rank_BGE-m3']}  MedCPT={r['rank_MedCPT']}  Gemma={r['rank_Gemma']}")
        print(f"  query : {r['query_text']}")
        print(f"  GOLD  : [{gold_pmid}] {short_title(gold_title, 90)}")
        print(f"        : \"{gold_sent}\"")
        if comp:
            print(f"  COMP  : [{comp_pmid}] {short_title(comp['title'], 90)}")
            print(f"        : \"{comp_sent}\"")


def show_ex2_candidates(pool: pl.DataFrame, n: int = 5) -> None:
    print(f"\n{'=' * 80}\nEx.2 — Query expansion  (top {n} of {len(pool)})\n{'=' * 80}")
    for r in pool.head(n).iter_rows(named=True):
        qid = r["qid"]
        gold_pmid = r["gold_pmids"][0]
        gold_title = r["gold_titles"][0]
        gold_abs = r["gold_abstracts"][0] or ""
        needles = r["added_syn"].split() if r["added_syn"] else None
        gold_sent = best_sentence(r["query_body"], gold_abs, must_include=needles)
        print(f"\nqid={qid}  body={r['rank_body']}  +syn={r['rank_+syn']}  "
              f"+long={r['rank_+long']}")
        print(f"  original  : {r['query_body']}")
        print(f"  + syn adds: {r['added_syn']}")
        print(f"  GOLD      : [{gold_pmid}] {short_title(gold_title, 90)}")
        print(f"            : \"{gold_sent}\"")


def show_ex3_candidates(pool: pl.DataFrame, n: int = 5) -> None:
    print(f"\n{'=' * 80}\nEx.3 — Full-text  (top {n} of {len(pool)})\n{'=' * 80}")
    for r in pool.head(n).iter_rows(named=True):
        qid = r["qid"]
        pmid = r["insufficient_pmids"][0]
        idx = r["gold_pmids"].index(pmid)
        title = r["gold_titles"][idx]
        abstract = r["gold_abstracts"][idx] or ""
        abs_sent = best_sentence(r["query_text"], abstract)
        body_text = ""
        for ch in top_chunks_for_pmid(qid, pmid, k=5):
            if ch.get("type") == "body" and ch.get("text"):
                body_text = ch["text"]
                break
        body_sent = best_sentence(r["query_text"], body_text)
        print(f"\nqid={qid}  abs={r['rank_abs_insuff']}  +chunks={r['rank_chk_insuff']}")
        print(f"  claim : {r['query_text']}")
        print(f"  GOLD  : [{pmid}] {short_title(title, 90)}")
        print(f"  abs   : \"{abs_sent}\"")
        print(f"  body  : \"{body_sent}\"")


def drill_ex2(qid: int) -> None:
    row = wide_ex2.filter(pl.col("qid") == qid).row(0, named=True)
    print("=" * 80)
    print(f"qid {qid}")
    print(f"  ORIGINAL: {row['query_body']}")
    print(f"  + SYN ADDS: {row['added_syn']}")
    print(
        f"  rank_body={row['rank_body']}  rank_+syn={row['rank_+syn']}  "
        f"rank_+long={row['rank_+long']}"
    )
    for pmid, title, abstract in zip(row["gold_pmids"], row["gold_titles"], row["gold_abstracts"]):
        print(f"  GOLD PMID {pmid}  |  {title}")
        needles = row["added_syn"].split() if row["added_syn"] else None
        s = best_sentence(row["query_body"], abstract or "", must_include=needles)
        print(f"    sentence-with-synonym: \"{s}\"")


# %% [markdown]
# ## 8. Manuscript-ready preview blocks
#
# Set the four `SELECTED_*` qids below, then run the next cell. Each example emits:
# 1. A markdown preview (easy to scan in the notebook), and
# 2. A LaTeX `tabular` snippet ready to paste into `paper.tex`.
#
# Ex.1 uses only three reranker columns (BM25 / MS-MARCO / Gemma) to keep the table
# narrow; BGE-m3 and MedCPT ranks are listed in the markdown header for reference.

# %%
SELECTED_EX1A: int | None = 393    # DymA / cleavage furrow — competitor (alpha-actinin paper) has matching phenotype location words but no DymA
SELECTED_EX1B: int | None = 1616   # AlxA <-> DdAlix — same gene, alternate name; BM25 cannot bridge
SELECTED_EX2: int | None = 313     # DetA -> DET1 — single canonical synonym appears in gold title
SELECTED_EX3: int | None = 534     # gefE / DIF sensitivity — body sentence near-paraphrases the claim

# Brainstorm: also render qid 535 (GefF / RasG) as an alternative Ex.1a whose competitor
# is itself a Ras-GEF paper (same gene family) — a more "biologically plausible mistake"
# than the DymA case (where the competitor is topically Dicty but a different protein class).
ALT_EX1A: int | None = 535

# Optional cleanup of curator-text artifacts in displayed queries (renderers only;
# the underlying retrieval ran on the original strings). Truncations use [...].
QUERY_OVERRIDES: dict[int, str] = {
    535: "GefF has GEF activity towards RasG [...]",  # original trailed: "unpublished results, cited in."
}

# Mechanism annotations rendered as a "Why" line under each example block.
WHY_NOTES: dict[int, str] = {
    393: (
        "MS-MARCO promotes a topically-related Dicty paper that shares the claim's "
        "phenotype-location vocabulary (cleavage furrow, phagocytic cup) but does not "
        "study DymA."
    ),
    535: (
        "MS-MARCO promotes a paper about Ras-superfamily GEFs (same gene family as the "
        "gold) but loses the specific GefF--RasG link the curator cited."
    ),
    1616: (
        "BM25 cannot bridge AlxA (curator's gene symbol) and DdAlix (the literature "
        "name for the same gene); Gemma matches them on semantic similarity."
    ),
}


def _ex1_competitor(row: dict, label: str) -> str | None:
    """For 'lost' we show what MS-MARCO promoted; for 'rescue' what BM25 promoted."""
    key = "top1_nongold_msmarco" if "lost" in label.lower() else "top1_nongold_bm25"
    return row.get(key)


def _display_query(qid: int, original: str) -> str:
    return QUERY_OVERRIDES.get(qid, original)


def render_ex1_markdown(qid: int, label: str) -> str:
    row = wide_ex1.filter(pl.col("qid") == qid).row(0, named=True)
    query_disp = _display_query(qid, row["query_text"])
    gold_pmid = row["gold_pmids"][0]
    gold_title = row["gold_titles"][0]
    gold_abs = lookup_article(gold_pmid)["abstract"] or ""
    gold_sent = best_sentence(row["query_text"], gold_abs)

    comp_pmid = _ex1_competitor(row, label)
    comp = lookup_article(comp_pmid) if comp_pmid else None
    comp_sent = best_sentence(row["query_text"], comp["abstract"] or "") if comp else ""
    c_ranks = (
        {n: rank_of_gold_for_qid(n, qid, [comp_pmid]) for n in runs_ex1}
        if comp_pmid else {}
    )

    why = WHY_NOTES.get(qid)
    # Two-column box: left = role (Query / Gold / Competitor / Why), right = content.
    # Title + snippet stacked inside the right cell via <br>.
    lines = [
        f"### Ex.1 — {label} &nbsp;(qid {qid})",
        "",
        "| | |",
        "|---|---|",
        f"| **Query** | {query_disp} |",
        (
            f"| **Gold** | PMID {gold_pmid} &nbsp; *{short_title(gold_title, 95)}*"
            f"<br>&nbsp;&nbsp;&nbsp;&nbsp;\"{gold_sent}\""
            f"<br>&nbsp;&nbsp;&nbsp;&nbsp;ranks: BM25 **{row['rank_BM25']}**, "
            f"MS-MARCO **{row['rank_MS-MARCO']}**, Gemma **{row['rank_Gemma']}** "
            f"(BGE-m3 {row['rank_BGE-m3']}, MedCPT {row['rank_MedCPT']}) |"
        ),
    ]
    if comp:
        lines.append(
            f"| **Competitor** | PMID {comp_pmid} &nbsp; *{short_title(comp['title'], 95)}*"
            f"<br>&nbsp;&nbsp;&nbsp;&nbsp;\"{comp_sent}\""
            f"<br>&nbsp;&nbsp;&nbsp;&nbsp;ranks: BM25 **{c_ranks['BM25']}**, "
            f"MS-MARCO **{c_ranks['MS-MARCO']}**, Gemma **{c_ranks['Gemma']}** |"
        )
    if why:
        lines.append(f"| **Why** | _{why}_ |")
    return "\n".join(lines) + "\n"


def _esc(s: str) -> str:
    """Minimal LaTeX escape for fields we paste into tabulars."""
    if s is None:
        return ""
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("~", "\\textasciitilde{}")
         .replace("^", "\\textasciicircum{}")
    )


def render_ex1_latex_block(qid: int, label: str) -> str:
    """Returns the inner rows for one example block (no \\begin{tabular})."""
    row = wide_ex1.filter(pl.col("qid") == qid).row(0, named=True)
    query_disp = _display_query(qid, row["query_text"])
    gold_pmid = row["gold_pmids"][0]
    gold_title = short_title(row["gold_titles"][0])
    gold_abs = lookup_article(gold_pmid)["abstract"] or ""
    gold_sent = best_sentence(row["query_text"], gold_abs)

    comp_pmid = _ex1_competitor(row, label)
    comp = lookup_article(comp_pmid) if comp_pmid else None
    comp_sent = best_sentence(row["query_text"], comp["abstract"] or "") if comp else ""
    c_ranks = (
        {n: rank_of_gold_for_qid(n, qid, [comp_pmid]) for n in runs_ex1}
        if comp_pmid else {}
    )
    comp_title = short_title(comp["title"]) if comp else ""

    lines = [
        f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\emph{{{_esc(label)}.}} "
        f"\\textit{{Query:}} {_esc(query_disp)}}} \\\\",
        f"Gold PMID {gold_pmid} --- {_esc(gold_title)} & "
        f"{row['rank_BM25']} & {row['rank_MS-MARCO']} & {row['rank_Gemma']} \\\\",
        f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\quad\\small\\itshape "
        f"``{_esc(gold_sent)}''}} \\\\",
    ]
    if comp:
        lines += [
            f"Competitor PMID {comp_pmid} --- {_esc(comp_title)} & "
            f"{c_ranks['BM25']} & {c_ranks['MS-MARCO']} & {c_ranks['Gemma']} \\\\",
            f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\quad\\small\\itshape "
            f"``{_esc(comp_sent)}''}} \\\\",
        ]
    why = WHY_NOTES.get(qid)
    if why:
        lines.append(
            f"\\multicolumn{{4}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\quad\\footnotesize "
            f"\\textit{{Why:}} {_esc(why)}}} \\\\"
        )
    return "\n".join(lines)


def render_ex1_latex(qids_labels: list[tuple[int, str]]) -> str:
    blocks = []
    for i, (qid, label) in enumerate(qids_labels):
        if i:
            blocks.append("\\midrule")
        blocks.append(render_ex1_latex_block(qid, label))
    body = "\n".join(blocks)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Qualitative examples illustrating reranker behavior on two curator-claim "
        "queries. Each block shows the gold article and one competitor article "
        "(the top-ranked non-gold article under the weaker reranker for 'Lexical signal lost', "
        "or under BM25 for 'Semantic rescue'). Ranks are positions of each article in the "
        "final ranking; lower is better. Quoted lines are best-matching sentences from "
        "each article's abstract.}\n"
        "\\label{tab:rerank_examples}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}p{8.0cm}rrr@{}}\n"
        "\\toprule\n"
        "Article & BM25 & MS-MARCO & Gemma \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def _gene_mapping(detected_genes: list[dict]) -> str:
    """Format detected gene(s) as 'curator-mention -> canonical (synonyms)'.

    The dictyBase pipeline stores `gene_name` as the canonical symbol and `synonyms`
    as the comma-separated list of alternative spellings; the curator typically uses
    one of the synonyms in the claim and the +syn expansion appends the canonical.
    """
    if not detected_genes:
        return ""
    parts = []
    for g in detected_genes:
        canonical = g.get("gene_name") or ""
        syns = g.get("synonyms") or ""
        if syns:
            parts.append(f"`{canonical}` (synonyms: {syns})")
        else:
            parts.append(f"`{canonical}`")
    return "; ".join(parts)


def render_ex2_markdown(qid: int) -> str:
    row = wide_ex2.filter(pl.col("qid") == qid).row(0, named=True)
    gold_pmid = row["gold_pmids"][0]
    gold_title = row["gold_titles"][0]
    gold_abs = row["gold_abstracts"][0] or ""
    needles = row["added_syn"].split() if row["added_syn"] else None
    gold_sent = best_sentence(row["query_body"], gold_abs, must_include=needles)
    gene_str = _gene_mapping(row.get("detected_genes") or [])
    # Extract the exact line appended to the query (everything after the original body).
    appended = ""
    if row["query_syn"] and row["query_syn"].startswith(row["query_body"]):
        appended = row["query_syn"][len(row["query_body"]):].strip()
    return (
        f"### Ex.2 — Gene-aware query expansion &nbsp;(qid {qid})\n\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| **Original query** | {row['query_body']} |\n"
        f"| **Detected gene** | {gene_str} |\n"
        f"| **Appended to query** | `{appended}` &nbsp;_(synonym `{row['added_syn']}` "
        f"attached to curator mention)_ |\n"
        f"| **Gold** | PMID {gold_pmid} &nbsp; *{short_title(gold_title, 100)}*"
        f"<br>&nbsp;&nbsp;&nbsp;&nbsp;\"{gold_sent}\" |\n"
        f"| **Rank of gold** | body **{row['rank_body']}** &nbsp;→&nbsp; "
        f"+syn **{row['rank_+syn']}** &nbsp;→&nbsp; "
        f"+syn\\&prod **{row['rank_+long']}** |\n"
    )


def render_ex2_latex(qid: int) -> str:
    row = wide_ex2.filter(pl.col("qid") == qid).row(0, named=True)
    gold_pmid = row["gold_pmids"][0]
    gold_title = short_title(row["gold_titles"][0], 95)
    gold_abs = row["gold_abstracts"][0] or ""
    needles = row["added_syn"].split() if row["added_syn"] else None
    gold_sent = best_sentence(row["query_body"], gold_abs, must_include=needles)
    appended = ""
    if row["query_syn"] and row["query_syn"].startswith(row["query_body"]):
        appended = row["query_syn"][len(row["query_body"]):].strip()
    # Format detected gene briefly: "curator-mention -> canonical".
    genes = row.get("detected_genes") or []
    gene_brief = ""
    if genes:
        g = genes[0]
        gene_brief = f"\\texttt{{{_esc(g.get('gene_name') or '')}}} (synonyms: " \
                     f"{_esc(g.get('synonyms') or '')})"
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Qualitative example illustrating gene-aware query expansion. "
        "The curator's claim mentions a gene by one of its alternative names; the "
        "expansion appends the canonical name (and optionally a product description) "
        "drawn from dictyBase metadata for the detected gene. Ranks are positions in "
        "the BGE-reranker-v2-m3 output on a fixed candidate pool.}\n"
        "\\label{tab:qe_example}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{@{}p{3.6cm}p{7.4cm}@{}}\n"
        "\\toprule\n"
        f"Original query & {_esc(row['query_body'])} \\\\\n"
        f"Detected gene & {gene_brief} \\\\\n"
        f"Appended to query & \\texttt{{{_esc(appended)}}} \\\\\n"
        f"Gold (PMID {gold_pmid}) & \\emph{{{_esc(gold_title)}}} \\\\\n"
        f"Gold sentence with synonym & \\small\\itshape ``{_esc(gold_sent)}'' \\\\\n"
        "\\midrule\n"
        "\\multicolumn{2}{@{}l@{}}{Rank of gold: "
        f"body = {row['rank_body']}, "
        f"+ synonyms = {row['rank_+syn']}, "
        f"+ syn \\& prod = {row['rank_+long']}.}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def render_ex3_markdown(qid: int) -> str:
    row = wide_ex3.filter(pl.col("qid") == qid).row(0, named=True)
    pmid = row["insufficient_pmids"][0]
    idx = row["gold_pmids"].index(pmid)
    title = row["gold_titles"][idx]
    abstract = row["gold_abstracts"][idx] or ""
    abs_sent = best_sentence(row["query_text"], abstract)
    body_text = ""
    for ch in top_chunks_for_pmid(qid, pmid, k=5):
        if ch.get("type") == "body" and ch.get("text"):
            body_text = ch["text"]
            break
    body_sent = best_sentence(row["query_text"], body_text)
    return (
        f"### Ex.3 — Full-text rescues abstract-insufficient claim &nbsp;(qid {qid})\n\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| **Curator claim** | {row['query_text']} |\n"
        f"| **Gold** | PMID {pmid} &nbsp; *{short_title(title, 100)}* |\n"
        f"| **Abstract (best sentence)** | _\"{abs_sent}\"_ |\n"
        f"| **Body chunk (best sentence)** | _\"{body_sent}\"_ |\n"
        f"| **Rank of gold** | abstract-only **{row['rank_abs_insuff']}** &nbsp;→&nbsp; "
        f"+chunks **{row['rank_chk_insuff']}** |\n"
    )


def render_ex3_latex(qid: int) -> str:
    row = wide_ex3.filter(pl.col("qid") == qid).row(0, named=True)
    pmid = row["insufficient_pmids"][0]
    idx = row["gold_pmids"].index(pmid)
    title = short_title(row["gold_titles"][idx], 95)
    abstract = row["gold_abstracts"][idx] or ""
    abs_sent = best_sentence(row["query_text"], abstract)
    body_text = ""
    for ch in top_chunks_for_pmid(qid, pmid, k=5):
        if ch.get("type") == "body" and ch.get("text"):
            body_text = ch["text"]
            break
    body_sent = best_sentence(row["query_text"], body_text)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Qualitative example illustrating the gain from full-text chunks on an "
        "abstract-insufficient claim. The cited abstract does not contain the specific cue "
        "the curator claim depends on; a body-text chunk of the same article does, and "
        "adding chunks to the corpus moves the article from outside the top-K into the top-K. "
        "Ranks are post-rerank positions under BGE-reranker-v2-m3.}\n"
        "\\label{tab:fulltext_example}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{@{}p{2.4cm}p{8.6cm}@{}}\n"
        "\\toprule\n"
        f"Curator claim & {_esc(row['query_text'])} \\\\\n"
        f"Gold PMID {pmid} & \\emph{{{_esc(title)}}} \\\\\n"
        f"Abstract (best sentence) & \\small\\itshape ``{_esc(abs_sent)}'' \\\\\n"
        f"Body chunk (best sentence) & \\small\\itshape ``{_esc(body_sent)}'' \\\\\n"
        "\\midrule\n"
        "\\multicolumn{2}{@{}l@{}}{Rank of gold under BGE-reranker-v2-m3: "
        f"abstract-only corpus = {row['rank_abs_insuff']}, "
        f"+ chunks = {row['rank_chk_insuff']}.}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# %% [markdown]
# ### Emit previews
#
# Markdown for the notebook (easy to scan); LaTeX for the manuscript.

# %%
def _hr(title: str) -> str:
    return f"\n---\n\n## {title}\n"


print(_hr("Ex.1a — Lexical signal lost (Option 2: keep qid 393, add Why annotation)"))
if SELECTED_EX1A is not None:
    print(render_ex1_markdown(SELECTED_EX1A, "Lexical signal lost"))

print(_hr("Ex.1a — Lexical signal lost (Option 1: swap to qid 535 with cleaned query + Why)"))
if ALT_EX1A is not None:
    print(render_ex1_markdown(ALT_EX1A, "Lexical signal lost"))

print(_hr("Ex.1b — Semantic rescue (qid 1616)"))
if SELECTED_EX1B is not None:
    print(render_ex1_markdown(SELECTED_EX1B, "Semantic rescue"))

print(_hr("Ex.2 — Gene-aware query expansion (qid 313)"))
if SELECTED_EX2 is not None:
    print(render_ex2_markdown(SELECTED_EX2))

print(_hr("Ex.3 — Full-text retrieval (qid 534)"))
if SELECTED_EX3 is not None:
    print(render_ex3_markdown(SELECTED_EX3))

# %%
# LaTeX output for paper.tex. Ex.1 packs both example blocks into one table.
ex1_pairs = []
if SELECTED_EX1A is not None:
    ex1_pairs.append((SELECTED_EX1A, "Lexical signal lost"))
if SELECTED_EX1B is not None:
    ex1_pairs.append((SELECTED_EX1B, "Semantic rescue"))
if ex1_pairs:
    print("% --- Ex.1: Reranker behavior ---")
    print(render_ex1_latex(ex1_pairs))
    print()
if SELECTED_EX2 is not None:
    print("% --- Ex.2: Query expansion ---")
    print(render_ex2_latex(SELECTED_EX2))
    print()
if SELECTED_EX3 is not None:
    print("% --- Ex.3: Full-text retrieval ---")
    print(render_ex3_latex(SELECTED_EX3))

# %%

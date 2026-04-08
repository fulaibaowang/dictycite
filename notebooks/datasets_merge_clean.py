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

# %%
import polars as pl
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import json
import os
from pathlib import Path


def _repo_root() -> Path:
    env = os.environ.get("DICTYCITE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    starts: list[Path] = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(Path.cwd().resolve())

    for start in starts:
        p = start
        while True:
            if (p / ".git").exists():
                return p
            if p.parent == p:
                break
            p = p.parent

    cwd = Path.cwd().resolve()
    for base in (cwd, cwd.parent, *cwd.parents):
        marker = base / "output" / "dicty_gold_build" / "1_curator_claims.parquet"
        if marker.is_file():
            return base
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def _output_gold_build() -> Path:
    return _repo_root() / "output" / "dicty_gold_build"


OUTPUT_GOLD_BUILD = _output_gold_build()

# %% [markdown]
# # citation-claims

# %%
claims = pl.read_parquet(OUTPUT_GOLD_BUILD / "1_curator_claims.parquet")
claims.head()

# %%
# look at a row closely
row = claims.row(1, named=True)
print(row)

# %%
# get preferred column set (same as before)
df = claims.select([
    "claim_plain",
    "anchors",
    "publication_ids",
    "citation_captions",
    "gene_id",
])

# %%
# clean up parentheses in citation_captions
df = df.with_columns(
    pl.col("citation_captions")
    .list.eval(
        pl.element()
        .str.replace_all(r"[()]", "")   # remove parentheses
        .str.replace_all(r"\s+", " ")   # collapse whitespace
        .str.strip_chars()              # <-- instead of .str.strip()
    )
    .alias("citation_captions")
)
# clean up comma
df = df.with_columns(
    pl.col("citation_captions")
    .list.eval(
        pl.element()
        .str.replace_all(r"[()]", "")
        .str.replace_all(r"\bet al\b\.?", "et al.")
        .str.replace_all(r",\s*et al\.", " et al.")
        .str.replace_all(r"et al\.\s*,\s*", "et al. ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    .alias("citation_captions")
)

# %%
# clean up very short claims, it shall have at least 6
min_words = 6

df2 = df.with_columns(
    # Count "words" by matching non-space sequences
    pl.col("claim_plain")
      .str.count_matches(r"\S+")
      .alias("n_words")
)

# 1) summary
summary = df2.select([
    pl.len().alias("rows_total"),
    (pl.col("n_words") < min_words).sum().alias("rows_to_drop"),
    (pl.col("n_words") >= min_words).sum().alias("rows_to_keep"),
])
print(summary)
# 2) inspect what you'd drop (optional)
to_drop = (
    df2.filter(pl.col("n_words") < min_words)
       .select(["n_words", "claim_plain", "gene_id"])
       .sort(["n_words", "claim_plain"])
)
with pl.Config(fmt_str_lengths=300):
    display(to_drop.head(50))

# 3) actually filter
df_filtered = df2.filter(pl.col("n_words") >= min_words).drop("n_words")

# %%
# let us examine the duplicated claims closely
# 1) merge duplicates by concatenating gene_id (unique + sorted)
merged = (
    df_filtered
    .group_by([
        "claim_plain",
        "anchors",
        "publication_ids",
        "citation_captions"
    ])
    .agg(
        pl.col("gene_id")
          .unique()
          .sort()
          .str.join(",")
          .alias("gene_id")
    )
    .select([
        "claim_plain",
        "anchors",
        "publication_ids",
        "citation_captions",
        "gene_id",
    ])
)

print("after merge:")
print("rows =", merged.height)
print("unique claim_plain =", merged.select(pl.col("claim_plain").n_unique()).item())

# 2) check if claim_plain is STILL duplicated (i.e., same claim text but different citation/anchors/etc.)
still_dups = (
    merged
    .with_columns(pl.len().over("claim_plain").alias("n"))
    .filter(pl.col("n") > 1)
    .sort(["claim_plain"])
)

print("still duplicated claim_plain rows =", still_dups.height)

# show them grouped together if any remain
still_dups

# %%
# with pl.Config(fmt_str_lengths=60):
#     display(still_dups)       


# %% [markdown]
# There are some duplicated claims.
#
# Except the first one, the others are claims with inconsistant citations, let us remove those except the first one.

# %%
# 1) add a stable row id to merged
merged_i = merged.with_row_count("row_nr")   # if this errors on your Polars, use: .with_row_index("row_nr")

# 2) recompute still_dups from merged_i (so it contains row_nr)
still_dups_i = (
    merged_i
    .with_columns(pl.len().over("claim_plain").alias("n"))
    .filter(pl.col("n") > 1)
    .sort(["claim_plain", "row_nr"])
)

# 3) drop everything in still_dups except the first row of still_dups
drop_row_nrs = still_dups_i.slice(1).select("row_nr")   # everything after the first row

merged_clean = (
    merged_i
    .join(drop_row_nrs, on="row_nr", how="anti")  # remove those rows
    .drop(["row_nr", "n"], strict=False)
)


# %%
merged_clean

# %%
# give claim ID, and we want to explode the list columns together to get one row per (claim_id, publication_id)
m = merged_clean.with_columns(
    pl.col("claim_plain").rank(method="dense").cast(pl.Int64).alias("claim_id")
).select([
    "claim_id",
    "claim_plain",
    "anchors",
    "publication_ids",
    "citation_captions",
    "gene_id"
])

# %%
# 2) sanity check: list columns must be aligned per row before explode (len(citation_captions) == len(publication_ids))
bad_rows = (
    m
    .with_columns([
        pl.col("publication_ids").list.len().alias("n_pub"),
        pl.col("citation_captions").list.len().alias("n_cap")
    ])
    .filter(
        (pl.col("n_pub") != pl.col("n_cap")) 
    )
    .select([
        "claim_id",
        "gene_id",
        "n_pub", "n_cap", 
        "claim_plain",
        "anchors",
        "publication_ids",
        "citation_captions"
    ])
    .sort(["n_pub", "n_cap"], descending=True)
)

print("Number of misaligned rows:", bad_rows.height)

with pl.Config(fmt_str_lengths=1000):
    display(bad_rows)

# %% [markdown]
# in the 4 cases here, len(publication_ids)>len(citation_captions), after closer look, it is the same author and same year, we manully make a patch here

# %%
m_fixed = (
    m.with_columns(
        pl.when(pl.col("claim_id") == 1534)
          .then(pl.lit(["Brandon et al. 1997a", "Brandon et al. 1997b"]))
        .when(pl.col("claim_id") == 471)
          .then(pl.lit(["Rupper et al. 2001a", "Rupper et al. 2001b"]))
        .when(pl.col("claim_id") == 1469)
          .then(pl.lit(["Pakes et al. 2012a", "Pakes et al. 2012b"]))
        .when(pl.col("claim_id") == 1195)
          .then(pl.lit(["Razeto et al. 2007a", "Razeto et al. 2007b"]))
        .otherwise(pl.col("citation_captions"))
        .alias("citation_captions")
    )
)

# %%
m_long = (
    m_fixed
    .explode(["publication_ids", "citation_captions"])
    .rename({"publication_ids": "publication_id"})
)

# %%
m_long

# %%
# now we exlpode anchors
# 1) Build mapping: (claim_id, publication_id) -> anchor_pos
anchor_map = (
    m_fixed
    .select(["claim_id", "anchors"])
    .explode("anchors")  # now each row is one struct {pos, pub_ids}
    .with_columns([
        pl.col("anchors").struct.field("pos").alias("anchor_pos"),
        pl.col("anchors").struct.field("pub_ids").alias("publication_id"),
    ])
    .explode("publication_id")  # one row per pub_id
    .select(["claim_id", "publication_id", "anchor_pos"])
)

# If there can be multiple anchor_pos for the same (claim_id, publication_id), keep them all:
anchor_map = (
    anchor_map
    .group_by(["claim_id", "publication_id"])
    .agg(pl.col("anchor_pos").sort().alias("anchor_pos"))
)

# 2) Join onto m_long
m_long2 = m_long.join(anchor_map, on=["claim_id", "publication_id"], how="left")

# 3) If you want exactly one row per (claim_id, publication_id, anchor_pos), explode anchor_pos:
m_long3 = m_long2.explode("anchor_pos")

# Final columns (example)
m_long3.select([
    "claim_id",
    "claim_plain",
    "anchor_pos",
    "publication_id",
    "citation_captions",
    "gene_id",
])

# %%
# now we get year from citation_captions
YEAR_RE = r"(18|19|20)\d{2}"

m_long3 = m_long3.with_columns(
    year=pl.col("citation_captions")
        .str.extract(YEAR_RE, 0)   # whole match
        .cast(pl.Int32)
)
m_long3

# %%
m_long4 = m_long3.select([
    "claim_id",
    "claim_plain",
    "anchors",        # keep if you still want the original struct list
    "anchor_pos",
    "publication_id",
    "citation_captions",
    "gene_id",
    "year",
])

# %% [markdown]
# i still see some claims needs cleaned up. They are almost identical with only dot diffrenece.
#

# %%
claim_key_expr = (
    pl.col("claim_plain")
    .str.to_lowercase()
    .str.replace_all(r"\.{2,}", ".")          # ".." / "..." -> "."
    .str.replace_all(r"[^a-z0-9\s]", " ")     # drop punctuation
    .str.replace_all(r"\s+", " ")             # collapse whitespace
    .str.strip_chars()
)

tmp = m_long4.with_columns(
    claim_key=claim_key_expr
)

# Optional: see groups where multiple raw claim_plain map to same key
near_same = (
    tmp.group_by("claim_key")
       .agg([
           pl.len().alias("n_rows"),
           pl.col("claim_plain").n_unique().alias("n_variants"),
           pl.col("claim_plain"),
       ])
       .filter(pl.col("n_variants") > 1)
       .sort("n_rows", descending=True)
)
near_same

# %% [markdown]
# yes, there are actually quite many near duplicates. shall be also merged

# %%
# 2) Create a new merged claim_id based on claim_key
tmp = tmp.with_columns(
    claim_id_new=pl.col("claim_key").rank(method="dense").cast(pl.Int64)
)

# 3) Merge rows safely and keep your column order
#    - choose a canonical claim_plain (first seen)
#    - merge gene_id as union joined by ","
claim_cleaned = (
    tmp
    .group_by([
        "claim_id_new",
        "publication_id",
        "citation_captions",
        "anchor_pos",
        "year",
    ])
    .agg([
        pl.first("claim_plain").alias("claim_plain"),
        pl.first("anchors").alias("anchors"),
        pl.col("gene_id").unique().sort().str.join(",").alias("gene_id"),
    ])
    .select([
        pl.col("claim_id_new").alias("claim_id"),
        "claim_plain",
        "anchors",
        "publication_id",
        "citation_captions",
        "gene_id",
        "anchor_pos",
        "year",
    ])
    .sort(["claim_id", "publication_id", "anchor_pos"])
)

claim_cleaned


# %%
year_summary = (
    claim_cleaned
    .with_columns(pl.col("year").fill_null(-1).alias("year2"))
    .group_by("year2")
    .agg(pl.len().alias("n"))
    .sort("year2")
    .with_columns(
        pl.when(pl.col("year2") == -1).then(None).otherwise(pl.col("year2")).alias("year")
    )
    .select(["year", "n"])
)

year_summary


# %%
# claim_cleaned.write_parquet(...)  # intermediate; dropped from output

# %%
claim_cleaned.select(
    pl.col("claim_id").n_unique().alias("n_unique_claim_id")
)


# %%
# claim_cleaned TSV export removed (intermediate; dropped from output)


# %% [markdown]
# I still see some duplicated claims. There are still a few edge cases. But at this point, I am not gonna trying to “perfectly” clean. I will try to pick up a golden set out of it.

# %%
# manual inspection, will pick some golden set maunally later
claim_cleaned_manual = claim_cleaned.filter(pl.col("anchor_pos") != 1)
# claim_cleaned_gold=...

# %% [markdown]
# # match pmid

# %%
pub_pmid = pl.read_csv(OUTPUT_GOLD_BUILD / "2_publication_id_pmid.csv")
pub_pmid

# %%
# check if all publcation id can be mapped to pmid

# pick the right df names
a = claim_cleaned
b = pub_pmid

# make sure both publication_id columns are the same type (I recommend Int64)
a_ids = a.select(pl.col("publication_id").cast(pl.Int64)).unique()
b_ids = b.select(pl.col("publication_id").cast(pl.Int64)).unique()

# overlap + only-in sets
overlap = a_ids.join(b_ids, on="publication_id", how="inner")
only_a  = a_ids.join(b_ids, on="publication_id", how="anti")
only_b  = b_ids.join(a_ids, on="publication_id", how="anti")

summary = pl.DataFrame({
    "set": ["claim_cleaned", "pub_pmid", "overlap", "only_claim_cleaned", "only_pub_pmid"],
    "unique_count": [a_ids.height, b_ids.height, overlap.height, only_a.height, only_b.height],
})

summary

# %% [markdown]
# 27 ids are not mapped.

# %%
claim_cleaned_pmid = (
    claim_cleaned
    .with_columns(pl.col("publication_id").cast(pl.Int64))
    .join(pub_pmid, on="publication_id", how="left")
    .with_columns(
        pl.col("pmid").fill_null("NA")  # or keep as null if you prefer
    )
)

claim_cleaned_pmid

# %%
# claim_cleaned_pmid.write_parquet(...)  # intermediate; dropped from output

claim_cleaned_pmid_nonNA = claim_cleaned_pmid.filter(
    pl.col("pmid").is_not_null() & (pl.col("pmid") != "NA")
)

# claim_cleaned_pmid_nonNA.write_parquet(...)  # intermediate; dropped from output


# %%
claim_cleaned_pmid_nonNA.select(
    pl.col("claim_id").n_unique().alias("n_unique_claim_id")
)


# %%
claim_cleaned_pmid_nonNA.select(
    pl.col("pmid").n_unique().alias("n_unique_pmid")
)

# %% [markdown]
# # how many we claims having abstracts on EPMC

# %%
EPMC = pl.read_parquet(OUTPUT_GOLD_BUILD / "3_articles_cleaned_abstract.parquet")

# %%
EPMC

# %%
a = claim_cleaned_pmid_nonNA.select(pl.col("pmid").cast(pl.Utf8).alias("pmid")).unique()
b = EPMC.select(pl.col("pmid").cast(pl.Utf8).alias("pmid")).unique()

overlap = a.join(b, on="pmid", how="inner")
only_a  = a.join(b, on="pmid", how="anti")
only_b  = b.join(a, on="pmid", how="anti")

summary = pl.DataFrame({
    "set": ["claim_cleaned_pmid_nonNA", "epmc_df", "overlap", "only_claim_cleaned_pmid", "only_epmc_df"],
    "unique_count": [a.height, b.height, overlap.height, only_a.height, only_b.height],
})

summary

# %%
only_claim_pmids = only_a.get_column("pmid")

rows_only_claim = claim_cleaned_pmid.filter(
    pl.col("pmid").cast(pl.Utf8).is_in(only_claim_pmids)
)

rows_only_claim

# %% [markdown]
# Some claims are not covered. It seems that most of them are old paper that there are no abstract available. So I remove those and merge title and abstract to the claim table

# %%
epmc_small = EPMC.select([
    pl.col("pmid").cast(pl.Utf8).alias("pmid"),
    pl.col("title").alias("title"),
    pl.col("abstract_clean").alias("abstract_clean"),
])

claim_small = claim_cleaned_pmid_nonNA.with_columns(
    pl.col("pmid").cast(pl.Utf8).alias("pmid")
)

# inner join = ignore only-claim PMIDs
claim_cleaned_pmid_nonNA_abstract = claim_small.join(epmc_small, on="pmid", how="inner")

claim_cleaned_pmid_nonNA_abstract

# %%
# claim_cleaned_pmid_nonNA_abstract.write_parquet(...)  # intermediate; dropped from output


# %%
claim_cleaned_pmid_nonNA_abstract.select(
    pl.col("claim_id").n_unique().alias("n_unique_claim_id")
)


# %%
claim_cleaned_pmid_nonNA_abstract.select(
    pl.col("pmid").n_unique().alias("n_unique_pmid")
)

# %% [markdown]
# # clean up even more and help for narrow down to a golden set

# %%
claims_u = (
    claim_cleaned_pmid_nonNA_abstract.select(["claim_id", "claim_plain"])
        .unique(subset=["claim_id"])
        .sort("claim_id")
)
claims_u.height

# %%
claims_u = claims_u.with_columns(
    claim_sim=(
        pl.col("claim_plain")
        # remove bracket chars
        .str.replace_all(r"[()\[\]{}]", "")
        # ; -> ,
        .str.replace_all(r";", ",")
        # collapse obvious runs
        .str.replace_all(r"\.{2,}", ".")
        .str.replace_all(r",(\s*,)+", ",")
        # fix mixed combos
        .str.replace_all(r",\s*\.", ".")
        .str.replace_all(r"\.\s*,", ".")
        # spacing rules
        .str.replace_all(r"\s+([,.])", r"$1")          # no space before , .
        .str.replace_all(r",([A-Za-z0-9])", r", $1")   # space after comma
        .str.replace_all(r"\.([A-Za-z])", r". $1")     # space after dot only before letters (avoids 3.14 -> 3. 14)
        # final whitespace cleanup
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
)


# %%
# # take a look
# (
#     claims_u
#     .write_csv("tmp/claims_u.tsv", separator="\t")
# )


# %% [markdown]
# Cluster near-duplicates with TF-IDF char ngrams

# %%
texts = claims_u.get_column("claim_sim").to_list()
ids   = claims_u.get_column("claim_id").to_list()
n = len(ids)

# --- TF-IDF (char ngrams) ---
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
X = vec.fit_transform(texts)

# --- Nearest neighbors (candidate pairs) ---
k = 30  #  <-----------lower value might miss pairs, larger values might slow down
nn = NearestNeighbors(n_neighbors=min(k, n), metric="cosine").fit(X)
dist, idx = nn.kneighbors(X, return_distance=True)

# --- Build candidate pair list with similarities ---
# (we'll inspect these, and also use them for clustering)
pairs = []
for i in range(n):
    for d, j in zip(dist[i], idx[i]):
        if j <= i:
            continue
        sim = 1.0 - float(d)
        pairs.append((i, j, sim))

pairs.sort(key=lambda x: x[2], reverse=True)

# Inspect top similar pairs (e.g. top 100)
topN = 10                               # <-------------------change here for look longer list
pairs_df = pl.DataFrame(
    {
        "i": [p[0] for p in pairs[:topN]],
        "j": [p[1] for p in pairs[:topN]],
        "sim": [p[2] for p in pairs[:topN]],
        "claim_id_i": [ids[p[0]] for p in pairs[:topN]],
        "claim_id_j": [ids[p[1]] for p in pairs[:topN]],
        "text_i": [texts[p[0]] for p in pairs[:topN]],
        "text_j": [texts[p[1]] for p in pairs[:topN]],
    }
).sort("sim", descending=True)

with pl.Config(fmt_str_lengths=500, tbl_rows=topN, tbl_cols=20):
    display(pairs_df)

# %% [markdown]
# I manually look at around the top 2000 results and a threshold of 
# sim_th = 0.65
# shall be fine

# %%
# -------------------------
# 1) CLUSTERING (union-find)
# -------------------------
sim_th = 0.6 #<------------I decided after maual inspection and iterate with later process in #4

parent = list(range(n))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

# union candidate pairs above threshold
for i, j, sim in pairs:
    if sim >= sim_th:
        union(i, j)

# assign group ids (1..K)
roots = [find(i) for i in range(n)]
root_to_gid = {}
gids = []
for r in roots:
    if r not in root_to_gid:
        root_to_gid[r] = len(root_to_gid) + 1
    gids.append(root_to_gid[r])

claim_group_map = pl.DataFrame({
    "claim_id": ids,
    "group_claim_id": gids,
})

# quick cluster size stats
cluster_sizes = claim_group_map.group_by("group_claim_id").len().sort("len", descending=True)
display(cluster_sizes.head(30))

# %%
# (4a_claim_groups.parquet is written after `rep` is defined — includes claim text + genes + canonical query.)

# %%
# ---------------------------------------------
# 2) TABLE: each group with variants in one row
# ---------------------------------------------
# Put variant claim_ids + variant texts into a list (for manual check)
group_variants = (
    claim_group_map
    .join(claims_u.select(["claim_id", "claim_plain", "claim_sim"]), on="claim_id", how="left")
    .group_by("group_claim_id")
    .agg([
        pl.col("claim_id").sort().alias("variant_claim_ids"),
        pl.col("claim_plain").alias("variant_claim_plain"),
        pl.col("claim_sim").alias("variant_claim_sim"),
        pl.len().alias("n_variants"),
    ])
    .sort(["n_variants", "group_claim_id"], descending=[True, False])
)

# Save for manual inspection
# TSV cannot store nested lists cleanly, so we stringify lists with join separators
group_variants_flat = (
    group_variants
    .with_columns([
        pl.col("variant_claim_ids")
          .list.eval(pl.element().cast(pl.Utf8)).list.join(",")
          .alias("variant_claim_ids"),
        pl.col("variant_claim_plain")
          .list.join(" ||| ")
          .alias("variant_claim_plain"),
        pl.col("variant_claim_sim")
          .list.join(" ||| ")
          .alias("variant_claim_sim"),
    ])
)

# group_variants_flat.write_csv("../tmp/group_variants.tsv", separator="\t")  # tmp/ removed
print("Wrote: tmp/group_variants.tsv")

# %%
# ------------------------------------------------------
# 3) Build a canonical representative text per group
#    (for downstream golden set + second TF-IDF peek)
# ------------------------------------------------------
# Choose canonical as the longest claim_plain (often most complete)
group_canon = (
    claim_group_map
    .join(claims_u.select(["claim_id", "claim_plain", "claim_sim"]), on="claim_id", how="left")
    .with_columns(pl.col("claim_plain").str.len_chars().alias("nchar"))
    .sort(["group_claim_id", "nchar"], descending=[False, True])
    .group_by("group_claim_id")
    .agg([
        pl.first("claim_plain").alias("canon_claim_plain"),
        pl.first("claim_sim").alias("canon_claim_sim"),
        pl.col("claim_id").sort().alias("variant_claim_ids"),
        pl.len().alias("n_variants"),
    ])
    .sort(["n_variants", "group_claim_id"], descending=[True, False])
)

# Save canonical table (flat)
group_canon_flat = group_canon.with_columns(
    pl.col("variant_claim_ids").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("variant_claim_ids")
)
# group_canon_flat.write_csv("../tmp/group_canonical.tsv", separator="\t")  # tmp/ removed
print("Wrote: tmp/group_canonical.tsv")

# %%
# ---------------------------------------------------------
# 4) PEEK #2: run TF-IDF again on canonical group texts
# ---------------------------------------------------------
group_texts = group_canon.get_column("canon_claim_sim").to_list()
group_ids   = group_canon.get_column("group_claim_id").to_list()
g = len(group_ids)

vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
X2 = vec2.fit_transform(group_texts)

k2 = 15
nn2 = NearestNeighbors(n_neighbors=min(k2, g), metric="cosine").fit(X2)
dist2, idx2 = nn2.kneighbors(X2, return_distance=True)

pairs2 = []
for i in range(g):
    for d, j in zip(dist2[i], idx2[i]):
        if j <= i:
            continue
        sim = 1.0 - float(d)
        pairs2.append((i, j, sim))
pairs2.sort(key=lambda x: x[2], reverse=True)

topN2 = 10
pairs2_df = pl.DataFrame({
    "i": [p[0] for p in pairs2[:topN2]],
    "j": [p[1] for p in pairs2[:topN2]],
    "sim": [p[2] for p in pairs2[:topN2]],
    "group_id_i": [group_ids[p[0]] for p in pairs2[:topN2]],
    "group_id_j": [group_ids[p[1]] for p in pairs2[:topN2]],
    "text_i": [group_texts[p[0]] for p in pairs2[:topN2]],
    "text_j": [group_texts[p[1]] for p in pairs2[:topN2]],
}).sort("sim", descending=True)

with pl.Config(fmt_str_lengths=500, tbl_rows=topN2, tbl_cols=20):
    display(pairs2_df)

# pairs2_df.write_csv("../tmp/group_pairwise_peek.tsv", separator="\t")  # tmp/ removed
print("Wrote: tmp/group_pairwise_peek.tsv")

# %%
claim_group_map.select(pl.col("group_claim_id").n_unique().alias("n_group_claim_id"))


# %% [markdown]
# we still have 1705 claims. This is nice

# %% [markdown]
# let us get our golden set

# %%
# --- 1) attach group ids to the long evidence table ---
gold_long = (
    claim_cleaned_pmid_nonNA_abstract
    .join(claim_group_map, on="claim_id", how="left")
    .join(claims_u.select(["claim_id", "claim_sim"]), on="claim_id", how="left")
)

# %%
# --- 2) pick one canonical query per group (longest claim_sim) ---
# canon_query = (
#     gold_long
#     .select(["group_claim_id", "claim_id", "claim_sim"])
#     .unique(subset=["group_claim_id", "claim_id"])
#     .with_columns(pl.col("claim_sim").str.len_chars().alias("nchar"))
#     .sort(["group_claim_id", "nchar"], descending=[False, True])
#     .group_by("group_claim_id")
#     .agg(pl.first("claim_sim").alias("query"))
# )
rep = (
    gold_long
    .select(["group_claim_id", "claim_id", "claim_sim"])
    .unique(subset=["group_claim_id", "claim_id"])   # one per variant
    .with_columns(pl.col("claim_sim").str.len_chars().alias("nchar"))
    .sort(["group_claim_id", "nchar"], descending=[False, True])
    .group_by("group_claim_id")
    .agg([
        pl.first("claim_id").alias("rep_claim_id"),
        pl.first("claim_sim").alias("query"),
    ])
)
rep

# %%
gene_by_claim = (
    claim_cleaned_pmid_nonNA_abstract
    .group_by("claim_id")
    .agg(
        pl.col("gene_id").unique().sort().str.join(",").alias("gene_id"),
    )
)
claim_groups_detail = (
    claim_group_map
    .join(claims_u, on="claim_id", how="left")
    .join(gene_by_claim, on="claim_id", how="left")
    .join(
        rep.select(["group_claim_id", "rep_claim_id", "query"]).rename({"query": "canonical_query"}),
        on="group_claim_id",
        how="left",
    )
    .with_columns(
        (pl.col("claim_id") == pl.col("rep_claim_id")).alias("is_representative_claim"),
    )
)
claim_groups_detail.write_parquet(OUTPUT_GOLD_BUILD / "4a_claim_groups.parquet")

# %%
# 3) Count variants (metadata) — from the mapping, not from citations
variant_counts = (
    claim_group_map
    .group_by("group_claim_id")
    .agg(pl.len().alias("n_variants"))
)

# 4) Keep ONLY the representative claim’s citations + anchors, then build golden set
rep_long = (
    gold_long
    .join(rep.select(["group_claim_id", "rep_claim_id"]), on="group_claim_id", how="inner")
    .filter(pl.col("claim_id") == pl.col("rep_claim_id"))
)

# 5) doc-level info (includes anchor_pos + citation_captions) for the representative query only
docs = (
    rep_long
    .group_by(["group_claim_id", "publication_id", "pmid", "title", "abstract_clean", "year"])
    .agg([
        pl.col("anchor_pos").drop_nulls().unique().sort().alias("anchor_pos"),
        pl.col("citation_captions").drop_nulls().unique().sort().alias("citation_captions"),
    ])
    .group_by("group_claim_id")
    .agg([
        pl.struct([
            "publication_id", "pmid", "title", "abstract_clean", "year",
            "anchor_pos", "citation_captions"
        ]).alias("docs"),
        pl.col("year").drop_nulls().unique().sort().alias("years"),   # years for THIS rep query only
    ])
)

gold = (
    rep
    .join(variant_counts, on="group_claim_id", how="left")
    .join(docs, on="group_claim_id", how="left")
    .with_columns([
        pl.col("docs").list.len().alias("n_citations"),
        pl.col("query").str.count_matches(r"\S+").alias("query_n_words"),
    ])
    .select([
        "group_claim_id",
        "rep_claim_id",
        "query",
        "n_variants",
        "n_citations",
        "query_n_words",
        "years",
        "docs",
    ])
)

gold.head(2)


# %%
gold.write_parquet(OUTPUT_GOLD_BUILD / "4b_golden_grouped.parquet")

# %%
gold_flat = (
    gold
    .explode("docs")
    .unnest("docs")
    .select([
        "group_claim_id",
        "query",
        "n_variants",
        "n_citations",
        "publication_id",
        "pmid",
        "title",
        "abstract_clean",
        "year",
    ])
)

# gold_flat.write_csv("../tmp/golden_flat.tsv", separator="\t")  # tmp/ removed
print("Wrote: tmp/golden_flat.tsv")

# %% [markdown]
# let us check out query statistics

# %%
# number of the citation a query have
cit_dist = gold.group_by("n_citations").len().sort("n_citations")
cit_dist


# %%
many_cit = gold.filter(pl.col("n_citations") > 5).select([
    "group_claim_id",
    "n_citations",
    "n_variants",
    "query",
])

with pl.Config(fmt_str_lengths=10_000, tbl_rows=500, tbl_cols=20):
    display(many_cit.sort("n_citations", descending=True))

# %%
rep.filter(pl.col("group_claim_id") == 1071).select(["group_claim_id", "rep_claim_id", "query"])
gene = (
    claim_cleaned_pmid_nonNA_abstract
    .filter(pl.col("claim_id") == 1835)
    .select(pl.col("gene_id").unique())
)

gene


# %% [markdown]
# Looked at http://dictybase.org/gene/DDB_G0277869, it is fine.
#

# %%
len_dist = (
    gold.with_columns(
        pl.when(pl.col("query_n_words") < 10).then(pl.lit("<10"))
          .when(pl.col("query_n_words") < 20).then(pl.lit("10-19"))
          .when(pl.col("query_n_words") < 40).then(pl.lit("20-39"))
          .otherwise(pl.lit("40+"))
          .alias("len_bucket")
    )
    .group_by("len_bucket")
    .len()
    .sort("len_bucket")
)
len_dist


# %%
var_dist = (
    gold.with_columns(
        pl.when(pl.col("n_variants") == 1).then(pl.lit("1"))
          .when(pl.col("n_variants") <= 3).then(pl.lit("2-3"))
          .when(pl.col("n_variants") <= 10).then(pl.lit("4-10"))
          .otherwise(pl.lit("11+"))
          .alias("variant_bucket")
    )
    .group_by("variant_bucket")
    .len()
    .sort("variant_bucket")
)

var_dist


# %%
# year
year_dist = (
    gold
    .select(["group_claim_id", "years"])
    .explode("years")
    .group_by("years")
    .len()
    .sort("years")
)
year_dist



# %% [markdown]
# conclusion of thses statistics:
# - 1-citation queries is domindated.
# - sentence length looks fine.
# - there are some big  vairant groups
#
# Thoughts:
# - This could be a good automatic dataset
# - If we should have a second track to manually pick a smaller gold subset
#     - this need to be very accurate
#     - in this case we shall keep oversample multi-citation groups, i.e those n_citations>1
# - those big variant_bucket might be helpful for stress testing / hard negatives (they are similar but different citations), for now variant_bucket is merged

# %%

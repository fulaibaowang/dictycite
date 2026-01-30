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
# #!/usr/bin/env python3
import os
import glob
import pyterrier as pt
import polars as pl
import pandas as pd


# %%
if not pt.started():
    pt.init()  # downloads Terrier jars, needs Java installed

# %%
PARQUET_PATH = "../output/cleaned/articles_all_cleaned_abstract.parquet"   
INDEX_PATH = os.path.abspath("../indexes/terrier_index_dicty_22.01.26")              

# %%
os.makedirs(INDEX_PATH, exist_ok=True)


# %%
def iter_docs_from_parquet(parquet_path: str):
    # keep the same script logic: dedup by pmid (keep last), and concatenate title + abstract
    df = (
        pl.read_parquet(parquet_path)
        .select(["pmid", "title", "abstract_clean"])
        .with_columns([
            pl.col("pmid").cast(pl.Utf8),
            pl.col("title").fill_null(""),
            pl.col("abstract_clean").fill_null(""),
        ])
    )

    latest = {}
    for pmid, title, abstract in df.iter_rows():
        pmid = (pmid or "").strip()
        if not pmid:
            continue
        latest[pmid] = {
            "docno": pmid,
            "text": f"{title} {abstract}".strip()
        }

    yield from latest.values()


# %%
# build index
indexer = pt.IterDictIndexer(
    INDEX_PATH,
    text_attrs=["text"],
    meta={"docno": 32, "text": 20000},  # ✅ allow long title+abstract
    overwrite=True,
    threads=1
)


# %%
# quick sanity:
first = next(iter_docs_from_parquet(PARQUET_PATH))
print(first.keys(), first["docno"][:10], first["text"][:80])

# %%
index_ref = indexer.index(iter_docs_from_parquet(PARQUET_PATH))
print("OK indexed at:", INDEX_PATH)

# %%
# check
index = pt.IndexFactory.of(index_ref)     # or pt.IndexFactory.of(INDEX_PATH)
coll_stats = index.getCollectionStatistics()

print("num_docs:", coll_stats.getNumberOfDocuments())
print("num_terms:", coll_stats.getNumberOfUniqueTerms())
print("num_tokens:", coll_stats.getNumberOfTokens())

# %%
# try query
bm25 = pt.terrier.Retriever(index, wmodel="BM25")
q = pd.DataFrame([{"qid":"q1", "query":"A recently reconstructed spatially fourth and temporally second order accurate, implicit, stable high order compact scheme has been employed to carry out simulations of the Oregonator model of excitable media."}])
bm25.transform(q).head(5)


# %%
res = bm25.transform(q)
top_pmid = str(res.loc[0, "docno"])   # res is your BM25 results df (pandas)
df = pl.read_parquet(PARQUET_PATH)

row = (
    df.with_columns(pl.col("pmid").cast(pl.Utf8))
      .filter(pl.col("pmid") == top_pmid)
      .select(["pmid", "title", "abstract_clean"])
)

row


# %% [markdown]
# results looks good!

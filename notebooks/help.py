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
# from a group claim id (query id) look up genes and claim text.
# `4a_claim_groups.parquet`: claim_id, group_claim_id, claim_plain, claim_sim, gene_id,
# rep_claim_id, canonical_query, is_representative_claim.

# %%
import polars as pl

claim_groups = pl.read_parquet("../output/dicty_gold_build/4a_claim_groups.parquet")
gold = pl.read_parquet("../output/dicty_gold_build/4b_golden_grouped.parquet")

# %%
claim_groups

# %%
gold.select(["group_claim_id", "query", "n_variants", "n_citations"]).head(10)

# %%

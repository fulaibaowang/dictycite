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
# from a group claim id (qurey id) this returns me the gene_id that i can take a look at dictybase. Makes my life easier.

# %%
import polars as pl
claim_cleaned_pmid_nonNA_abstract=pl.read_parquet("../output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet")

# %%
claim_cleaned_pmid_nonNA_abstract

# %%
claim_group_map=pl.read_csv("../output/cleaned/claim_group_map.tsv", separator="\t")

# %%
group_gene_map = (
    claim_cleaned_pmid_nonNA_abstract
    .select(["claim_id", "gene_id"])
    .unique()
    .join(claim_group_map, on="claim_id", how="left")
    .group_by("group_claim_id")
    .agg(
        pl.col("gene_id")
          .drop_nulls()
          .unique()
          .sort()
          .alias("gene_ids")
    )
)

# %%
group_gene_map

# %%
group_gene_dict = {
    row[0]: row[1]
    for row in group_gene_map.select(["group_claim_id", "gene_ids"]).iter_rows()
}

def get_gene_ids(group_claim_id):
    return group_gene_dict.get(group_claim_id, [])



# %%
get_gene_ids(1468)


# %% [markdown]
# http://dictybase.org/gene/DDB_G0284977

# %%
get_gene_ids(1616)


# %%
get_gene_ids(1005)



# %%

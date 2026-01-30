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
# # get all publications for one gene

# %% [markdown]
# dictbase have two links like:
#
# http://dictybase.org/gene/DDB_G0283907/references.json 
#
# http://dictybase.org/gene/DDB_G0283907/gene/references.json
#
# the first one have full list and from that I can get internal publication ID and pmid pair

# %%
import json
import re
from typing import Any, Iterable

import requests
import polars as pl

# %%
BASE = "http://dictybase.org"

PUB_RE = re.compile(r"/publication/(\d+)")
PMID_RE = re.compile(
    r"(?:/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)|view\.ncbi\.nlm\.nih\.gov/pubmed/(\d+)"
)

def _find_records_lists(node: Any) -> Iterable[list]:
    """Yield every list found at any nested key named 'records'."""
    if isinstance(node, dict):
        recs = node.get("records")
        if isinstance(recs, list):
            yield recs
        for v in node.values():
            yield from _find_records_lists(v)
    elif isinstance(node, list):
        for v in node:
            yield from _find_records_lists(v)

def _clean_url(u: str) -> str:
    """
    Some entries have concatenated duplicate URLs like:
    'https://...https://...'
    Keep the first URL if it looks duplicated.
    """
    if not u:
        return u
    # If two 'http' appear, keep up to the second occurrence
    first = u.find("http")
    second = u.find("http", first + 4)
    if first == 0 and second > 0:
        return u[:second]
    return u

def fetch_gene_pubid_pmid(gene_id: str, session: requests.Session | None = None) -> pl.DataFrame:
    close_session = False
    if session is None:
        session = requests.Session()
        close_session = True

    try:
        url = f"{BASE}/gene/{gene_id}/references.json"
        r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        payload = json.loads(r.text)

        rows = []
        for records in _find_records_lists(payload):
            for rec in records:
                if not isinstance(rec, dict):
                    continue

                ref_links = rec.get("ref_link") or []
                pub_id = None
                pmid = None

                for it in ref_links:
                    if not isinstance(it, dict):
                        continue
                    u = _clean_url((it.get("url") or "").strip())
                    if not u:
                        continue

                    m = PUB_RE.search(u)
                    if m:
                        pub_id = m.group(1)

                    m = PMID_RE.search(u)
                    if m:
                        pmid = m.group(1) or m.group(2)

                if pub_id or pmid:
                    rows.append(
                        {"gene_id": gene_id, "publication_id": pub_id, "pmid": pmid}
                    )

        df = (
            pl.from_dicts(rows)
            if rows
            else pl.DataFrame({"gene_id": [], "publication_id": [], "pmid": []})
        )

        return (
            df.with_columns(
                pl.col("gene_id").cast(pl.Utf8),
                pl.col("publication_id").cast(pl.Utf8),
                pl.col("pmid").cast(pl.Utf8),
            )
            .unique(subset=["gene_id", "publication_id", "pmid"])
            .sort(["gene_id", "publication_id", "pmid"])
        )

    finally:
        if close_session:
            session.close()

def fetch_many_genes_pubid_pmid(gene_ids: list[str]) -> pl.DataFrame:
    with requests.Session() as s:
        dfs = [fetch_gene_pubid_pmid(g, session=s) for g in gene_ids]
    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()

# ---- example ----
df = fetch_gene_pubid_pmid("DDB_G0283907")
print(df)

# ---- multiple genes example ----
# gene_ids = ["DDB_G0283907", "DDB_G0277809"]
# df_all = fetch_many_genes_pubid_pmid(gene_ids)
# df_all.write_csv("gene_publication_pmid.tsv", separator="\t")

# %% [markdown]
# some gene pages  don’t include a “References” tab, so we can skip them

# %%
# --- extract refs from references.json (same as before) ---
PUB_RE = re.compile(r"/publication/(\d+)")
PMID_RE = re.compile(
    r"(?:/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)|view\.ncbi\.nlm\.nih\.gov/pubmed/(\d+)"
)

def _find_records_lists(node):
    if isinstance(node, dict):
        recs = node.get("records")
        if isinstance(recs, list):
            yield recs
        for v in node.values():
            yield from _find_records_lists(v)
    elif isinstance(node, list):
        for v in node:
            yield from _find_records_lists(v)

def _clean_url(u: str) -> str:
    if not u:
        return u
    first = u.find("http")
    second = u.find("http", first + 4)
    if first == 0 and second > 0:
        return u[:second]
    return u

def extract_pubid_pmid_from_references_payload(gene_id: str, payload) -> pl.DataFrame:
    rows = []
    seen = set()

    for records in _find_records_lists(payload):
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ref_links = rec.get("ref_link") or []
            if not isinstance(ref_links, list):
                continue

            pub_id = None
            pmid = None
            for it in ref_links:
                if not isinstance(it, dict):
                    continue
                u = _clean_url((it.get("url") or "").strip())
                if not u:
                    continue
                m = PUB_RE.search(u)
                if m:
                    pub_id = m.group(1)
                m = PMID_RE.search(u)
                if m:
                    pmid = m.group(1) or m.group(2)

            if not pub_id and not pmid:
                continue

            key = (pub_id, pmid)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"gene_id": gene_id, "publication_id": pub_id, "pmid": pmid})

    df = pl.from_dicts(rows) if rows else pl.DataFrame({"gene_id": [], "publication_id": [], "pmid": []})
    return df.with_columns(
        pl.col("gene_id").cast(pl.Utf8),
        pl.col("publication_id").cast(pl.Utf8),
        pl.col("pmid").cast(pl.Utf8),
    ).unique()

# --- detect whether gene page has references tab and where it points ---
CONFIG_RE = re.compile(r"var\s+config\s*=\s*(\[\{.*?\}\]);\s*var\s+panel", re.DOTALL)

def get_references_source_from_gene_page(gene_id: str, session: requests.Session) -> str | None:
    """
    Return the references.json URL path (e.g. "/gene/DDB_G0283907/references.json")
    if the gene page declares a References tab, else None.
    """
    url = f"{BASE}/gene/{gene_id}"
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    html = r.text

    m = CONFIG_RE.search(html)
    if not m:
        return None

    config_json = m.group(1)
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        return None

    # config[0]["items"] contains tabs
    try:
        tabs = config[0]["items"]
    except Exception:
        return None

    for tab in tabs:
        if not isinstance(tab, dict):
            continue
        # common markers
        if tab.get("key") == "references":
            return tab.get("source")
        src = tab.get("source") or ""
        if isinstance(src, str) and src.endswith("/references.json"):
            return src

    return None

def fetch_gene_references_if_present(gene_id: str) -> tuple[str, pl.DataFrame]:
    """
    Single-gene test:
      - checks if References tab exists
      - if yes, fetches references.json and returns extracted table
      - if no, returns empty df
    """
    with requests.Session() as s:
        src = get_references_source_from_gene_page(gene_id, s)
        if not src:
            return "NO_REFERENCES_TAB", pl.DataFrame({"gene_id": [], "publication_id": [], "pmid": []})

        # src is usually like "/gene/DDB_G0283907/references.json"
        ref_url = src if src.startswith("http") else (BASE + src)
        r = s.get(ref_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code == 404:
            return f"REFERENCES_TAB_BUT_404 ({ref_url})", pl.DataFrame({"gene_id": [], "publication_id": [], "pmid": []})
        r.raise_for_status()
        payload = r.json()

    df = extract_pubid_pmid_from_references_payload(gene_id, payload)
    return f"OK ({ref_url})", df

# ---- try your example gene ----
status, df = fetch_gene_references_if_present("DDB_G0272226")
print(status)
print(df)

# ---- try a known gene with references ----
# status2, df2 = fetch_gene_references_if_present("DDB_G0283907")
# print(status2)
# print(df2.head(5))


# %%
# ---- try a known gene with references ----
status2, df2 = fetch_gene_references_if_present("DDB_G0283907")
print(status2)
print(df2)

# %% [markdown]
# # full result

# %% [markdown]
# command: docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 fulaibaowang/dictycite:16.01.2026 python /dictycite/dicty_publication.py --limit 0   --timeout 90 --sleep-base 0.35 --sleep-jitter 0.10 --out "/dictycite/output/gene_publication_pmid.parquet"
#
# takes around 10 hrs

# %% [markdown]
# In the first run some gene ids were not ot accessible at the moment, so were refetched in a second run. Merge two results and extract the publication IDs and pmids.

# %%
df = pl.read_parquet("output/gene_publication_pmid.parquet")
df

# %%
df_rerun = pl.read_parquet("output/gene_publication_pmid_rerun_failed.parquet")


# %%
df_clean = (
    df
    .filter(pl.col("publication_id").is_not_null())
    .unique()
)

df_rerun_clean = (
    df_rerun
    .filter(pl.col("publication_id").is_not_null())
    .unique()
)

# %%
rerun_gene_ids = df_rerun_clean.select("gene_id").drop_nulls().unique()
rerun_gene_ids

# %%
df_merged = (
    pl.concat([df_clean, df_rerun_clean], how="vertical")
    .unique()
)

pub_pmid = (
    df_merged
    .select(["publication_id", "pmid"])
    .unique()
)

pub_pmid_sorted = (
    pub_pmid
    .with_columns([
        pl.col("publication_id").cast(pl.Int64),
        pl.col("pmid").cast(pl.Int64),
    ])
    .sort("publication_id")
)



# %%
pub_pmid_sorted

# %%
pub_pmid_sorted.write_csv("output/publication_id_pmid.csv")


# %% [markdown]
# ## check if we cover the ids that in the curotor notes

# %%
publications_in_curated_notes = pl.read_parquet("output/publications.parquet")
publications_in_curated_notes

# %% [markdown]
# A quick to see if pmid correctly corrspond caption_plain and year in our table, e.g https://pubmed.ncbi.nlm.nih.gov/12600317/

# %%
pub_pmid_sorted.filter(
    pl.col("publication_id").is_in([2046, 6732,1793])
)


# %% [markdown]
# And yes, it is.
#
# Let see if all publication_id in curator notes get covered.

# %%
pub_ids = pub_pmid_sorted.select("publication_id").unique()
note_ids = publications_in_curated_notes.select("publication_id").unique()
overlap = pub_ids.join(note_ids, on="publication_id", how="inner")
only_publications = pub_ids.join(note_ids, on="publication_id", how="anti")
only_notes = note_ids.join(pub_ids, on="publication_id", how="anti")


# %%
only_notes

# %% [markdown]
# only 29 publication do not link to a pmid. That shall be fine!

# %%

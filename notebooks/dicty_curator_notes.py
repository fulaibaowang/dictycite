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
import requests
import time
import random
from bs4 import BeautifulSoup
import polars as pl
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


# %%
BASE = "http://dictybase.org"

# %% [markdown]
# # get curator notes

# %%
# good for large qureies
session = requests.Session()
session.headers.update({
    "User-Agent": "dictybase-curator-notes/0.1"
})


# %%
def get_curator_notes_html(gene_id: str, timeout: float = 15.0) -> str | None:
    """
    Return curator notes as HTML-ish string (with <i>, <br>, etc.),
    or None if 404 / no notes.
    """
    url = f"{BASE}/gene/{gene_id}/gene/summary.json"
    r = session.get(url, timeout=timeout)

    if r.status_code == 404:
        return None
    r.raise_for_status()

    data = r.json()

    try:
        col0 = data[0]["items"][0]
        col_items = col0["content"][0]["items"]
        content_row = col_items[1]                  # after "Curator Notes" title
        tokens = content_row["content"][0]["items"]
    except (KeyError, IndexError, TypeError):
        return None

    fragments = []
    for t in tokens:
        if "text" in t:
            fragments.append(t["text"])
        elif "caption" in t:
            fragments.append(t["caption"])

    html = "".join(fragments).strip()
    return html or None


def get_curator_notes_plain(gene_id: str, timeout: float = 15.0) -> str | None:
    """
    Plain-text version of curator notes (HTML stripped).
    """
    html = get_curator_notes_html(gene_id, timeout=timeout)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


# %%
def get_curator_notes_tokens(gene_id: str, timeout: float = 15.0):
    url = f"{BASE}/gene/{gene_id}/gene/summary.json"
    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()

    try:
        col0 = data[0]["items"][0]
        col_items = col0["content"][0]["items"]
        content_row = col_items[1]
        tokens = content_row["content"][0]["items"]
    except (KeyError, IndexError, TypeError):
        return None

    return [t for t in tokens if isinstance(t, dict)]



# %%
SENT_TOKEN_PAT = re.compile(r"\([^)]*\)|\[\[PUB:(\d+)\]\]")  # parenthetical OR marker

def _clean_text(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def _build_claim_and_anchors_from_marker_sentence(sent_markers: str):
    """
    From a sentence containing [[PUB:####]] markers:
      - remove citation parentheticals (those containing markers)
      - remove bare markers
      - record anchor positions (char offsets) in the cleaned claim text
    """
    raw_claim_parts = []
    anchors_raw = {}  # pos_raw -> [pub_ids]
    i = 0

    for m in SENT_TOKEN_PAT.finditer(sent_markers):
        start, end = m.start(), m.end()
        before = sent_markers[i:start]
        raw_claim_parts.append(before)

        chunk = sent_markers[start:end]

        # parenthetical
        if chunk.startswith("(") and chunk.endswith(")"):
            pub_ids = PUB_MARKER_PAT.findall(chunk)
            if pub_ids:
                pos_raw = len("".join(raw_claim_parts))
                anchors_raw.setdefault(pos_raw, []).extend(pub_ids)
                # drop the entire citation parenthetical
            else:
                raw_claim_parts.append(chunk)  # keep non-citation parentheses
        else:
            # bare marker [[PUB:####]]
            pid = m.group(1)
            if pid:
                pos_raw = len("".join(raw_claim_parts))
                anchors_raw.setdefault(pos_raw, []).append(pid)
            # drop the marker

        i = end

    raw_claim_parts.append(sent_markers[i:])
    raw_claim = "".join(raw_claim_parts)

    claim_plain = _clean_text(raw_claim)

    # remap raw anchor positions -> cleaned-text positions
    anchors = []
    for pos_raw, pids in anchors_raw.items():
        prefix_clean = _clean_text(raw_claim[:pos_raw])
        pos_clean = len(prefix_clean)
        anchors.append({
            "pos": pos_clean,
            "pub_ids": [int(x) for x in _dedup_keep_order(pids)],
        })

    anchors = sorted(anchors, key=lambda d: d["pos"])
    return claim_plain, anchors



# %%

YEAR_PAT = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
PUB_URL_PAT = re.compile(r"^/publication/(\d+)\b")
PUB_MARKER_PAT = re.compile(r"\[\[PUB:(\d+)\]\]")

def _strip_html_to_plain(s: str) -> str:
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

def _dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _is_citation_only_sentence(sent: str) -> bool:
    """
    True if sentence contains PUB markers but otherwise has no 'real' content
    (only parentheses/punctuation/spaces).
    """
    if not PUB_MARKER_PAT.search(sent):
        return False

    # Remove markers, then check what's left
    s = PUB_MARKER_PAT.sub("", sent)
    # Remove punctuation/whitespace
    s = re.sub(r"[\s\(\)\[\]\{\},;:.!?\-–—]+", "", s)
    return s == ""  # nothing meaningful left

def _remove_citations_from_sentence(sent_with_markers: str) -> str:
    """
    Remove citation markers and also remove citation-only parentheticals.
    Keep other parentheticals like (pkaC, pkaR).
    """
    def repl_paren(m):
        block = m.group(0)  # "( ... )"
        inner = block[1:-1]
        if not PUB_MARKER_PAT.search(inner):
            return block  # not a citation paren
        # remove markers
        inner2 = PUB_MARKER_PAT.sub("", inner)
        # if only punctuation/whitespace remains, drop entire parentheses
        inner2_chk = re.sub(r"[\s,;:.]+", "", inner2)
        if inner2_chk == "":
            return ""
        # otherwise keep remaining text (rare "see ..." cases)
        inner2 = inner2.strip()
        return f"({inner2})"

    # 1) first handle parenthetical blocks that contain markers
    s = re.sub(r"\([^)]*\)", repl_paren, sent_with_markers)

    # 2) remove any remaining markers not inside parentheses
    s = PUB_MARKER_PAT.sub("", s)

    # 3) cleanup spacing
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

def extract_publication_claims_from_tokens(gene_id: str, tokens):
    if not tokens:   # None or empty list
        return [], []
    parts = []
    pub_meta = {}  # pub_id -> {caption_plain, year}

    for t in tokens:
        if "text" in t:
            parts.append(str(t["text"]))
            continue

        caption = t.get("caption")
        url = t.get("url")
        if caption is None:
            continue

        caption_str = str(caption)
        m = PUB_URL_PAT.match(str(url)) if url else None

        if m:
            pub_id = m.group(1)
            parts.append(f"[[PUB:{pub_id}]]")

            cap_plain = _strip_html_to_plain(caption_str)
            years = YEAR_PAT.findall(cap_plain)
            year = int(years[-1]) if years else None
            pub_meta[pub_id] = {"caption_plain": cap_plain, "year": year}
        else:
            parts.append(caption_str)

    html_with_markers = "".join(parts)

    tmp_plain = _strip_html_to_plain(
        html_with_markers.replace("<br>", ". ").replace("<br/>", ". ")
    )

    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", tmp_plain) if s.strip()]

    merged = []
    for s in raw:
        if merged and _is_citation_only_sentence(s):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)

    claims = []
    for sent_markers in merged:
        pub_ids = PUB_MARKER_PAT.findall(sent_markers)
        if not pub_ids:
            continue

        pub_ids_u = _dedup_keep_order(pub_ids)

        # anchors + claim_plain (no citations)
        claim_plain, anchors = _build_claim_and_anchors_from_marker_sentence(sent_markers)

        # sentence_plain (citations rendered), plus a marked version
        sent_plain = sent_markers
        cited_sentence_marked = sent_markers

        caps = []
        years = []
        for pid in pub_ids_u:
            meta = pub_meta.get(pid, {})
            cap_plain = meta.get("caption_plain", f"publication/{pid}")

            sent_plain = sent_plain.replace(f"[[PUB:{pid}]]", cap_plain)
            cited_sentence_marked = cited_sentence_marked.replace(f"[[PUB:{pid}]]", f"[CITE:{pid}]")

            caps.append(cap_plain)
            if meta.get("year") is not None:
                years.append(int(meta["year"]))

        claims.append({
            "gene_id": gene_id,
            "sentence_markers": sent_markers,                 # keep original marker placement
            "sentence_plain": sent_plain,                     # readable with author-year
            "cited_sentence_marked": cited_sentence_marked,   # [CITE:####] markers
            "claim_plain": claim_plain,                       # claim only
            "anchors": anchors,                               # [{"pos": int, "pub_ids":[...]}]
            "publication_ids": [int(x) for x in pub_ids_u],
            "citation_captions": _dedup_keep_order(caps),
            "citation_years": _dedup_keep_order(years),
        })

    # publication table rows (dedup later)
    pub_rows = []
    for pid, meta in pub_meta.items():
        pub_rows.append({
            "publication_id": int(pid),
            "caption_plain": meta.get("caption_plain"),
            "year": meta.get("year"),
        })

    return claims, pub_rows



# %%
# Example
gid = "DDB_G0283907"
print("HTML version:\n", get_curator_notes_html(gid)[:300], "...\n")
print("Plain text:\n", get_curator_notes_plain(gid)[:300], "...")

# %%
gid = "DDB_G0283907"
tokens = get_curator_notes_tokens(gid)

claims, pubs = extract_publication_claims_from_tokens(gid, tokens)

claims_df = pl.DataFrame(claims)
pub_df = pl.DataFrame(pubs).unique(subset=["publication_id"])

claims_df.select(["gene_id","claim_plain","anchors","publication_ids"]).head(5)


# %%
row0 = claims_df.row(0, named=True)
for k, v in row0.items():
    print(f"\n=== {k} ===\n{v}")


# %%
# Example empty
gid = "DDB_G3946984"
note = get_curator_notes_html(gid)

if note is None:
    print(f"{gid}: no curator notes (404 or empty)")
else:
    print("HTML version:\n", note[:300], "...\n")
tokens = get_curator_notes_tokens(gid)

claims, pubs = extract_publication_claims_from_tokens(gid, tokens)

claims_df = pl.DataFrame(claims)
if pubs:  # non-empty list
    pub_df = pl.DataFrame(pubs).unique(subset=["publication_id"])
else:
    pub_df = pl.DataFrame(schema={"publication_id": pl.Int64, "caption_plain": pl.Utf8, "year": pl.Int32})

claims_df.head(), pub_df


# %% [markdown]
# # get full protein list

# %%
# List of Reviews and associated genes (Updated monthly)
df_review = pl.read_csv("dictybase_files/Reviews.txt", separator="\t", has_header=False)
df_review.head()

# %%
genes_review = df_review.select(df_review.columns[0]).unique()
len(genes_review)

# %%
# DDB_G curation status (Updated monthly)
df_status = pl.read_csv("dictybase_files/DDB_G-curation_status.txt", separator="\t", has_header=False,truncate_ragged_lines=True)
df_status.head()

# %%
genes_status = df_status.select(df_status.columns[0]).unique()
len(genes_status)

# %%
genes_status_nonempty = (
    df_status
    .filter(
        pl.col(df_status.columns[1]).is_not_null()
        & (pl.col(df_status.columns[1]) != "")
    )
    .select(pl.col(df_status.columns[0]))
    .unique()
)
len(genes_status_nonempty)

# %%
#dictyBase ID, gene names, synonyms, and gene products (Updated monthly)
df_gene = pl.read_csv("dictybase_files/gene_information.txt", separator="\t", has_header=True)
df_gene.head()

# %%
genes_gene = df_gene.select(df_gene.columns[0]).unique()
len(genes_gene)

# %%
#DDB-DDB_G-UniProt mapping (Updated monthly)
df_mapping = pl.read_csv("dictybase_files/DDB-GeneID-UniProt.txt", separator="\t", has_header=True)
df_mapping.head()

# %%
genes_mapping = df_mapping.select(df_mapping.columns[1]).unique()
len(genes_mapping)

# %%
print(
    df_review.height,
    genes_status.height,
    genes_gene.height,
    genes_mapping.height,
)


# %%
s = pl.concat([
    genes_status.to_series(0),
    genes_gene.to_series(0),
    genes_mapping.to_series(0),
])

s.unique().len()

# %% [markdown]
# DDB_G curation status file contains all gene id

# %% [markdown]
# # loop to get all curated notes

# %%
genes_status.head()


# %%
def polite_sleep(base=0.2, jitter=0.10):
    time.sleep(base + random.random() * jitter)


# %%
rows = []
for gid in genes_status.head(20).to_series():  #<-----------------remove head for extract all
    html = None
    plain = None
    try:
        html = get_curator_notes_html(gid)
        if html:
            plain = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except requests.RequestException as e:
        print(f"{gid}: failed ({e})")

    rows.append({
        "gene_id": gid,
        "curator_notes_html": html,
        "curator_notes_plain": plain,
    })

    polite_sleep()

# %%
rows

# %%
#Optional (but recommended): incremental save / resume
from pathlib import Path

OUT = Path("curator_notes.parquet")

if OUT.exists():
    df_done = pl.read_parquet(OUT)
    done_ids = set(df_done["gene_id"].to_list())
    print(f"Resuming: {len(done_ids)} genes already processed")
else:
    df_done = None
    done_ids = set()
    print("Starting fresh")

def polite_sleep(base=0.15, jitter=0.10):
    time.sleep(base + random.random() * jitter)

rows_buffer = []
BATCH_SIZE = 200   # write every 200 genes

for gid in genes_status.head(100).select("column_1").to_series():   #<-----------------remove head for extract all
    if gid in done_ids:
        continue   # ← THIS enables resume

    try:
        html = get_curator_notes_html(gid)
        plain = (
            BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            if html else None
        )
    except Exception as e:
        print(f"{gid}: failed ({e})")
        html = None
        plain = None

    rows_buffer.append({
        "gene_id": gid,
        "curator_notes_html": html,
        "curator_notes_plain": plain,
    })

    # periodically flush to disk
    if len(rows_buffer) >= BATCH_SIZE:
        df_batch = pl.DataFrame(rows_buffer)

        if OUT.exists():
            df_batch.write_parquet(OUT, append=True)
        else:
            df_batch.write_parquet(OUT)

        rows_buffer.clear()

    polite_sleep()
    
if rows_buffer:
    df_batch = pl.DataFrame(rows_buffer)
    if OUT.exists():
        df_batch.write_parquet(OUT, append=True)
    else:
        df_batch.write_parquet(OUT)

# %%
df = pl.read_parquet("curator_notes.parquet")
df

# %% [markdown]
# # full results

# %% [markdown]
# command: docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 fulaibaowang/dictycite:22.12.2025 python dicty_curator_notes.py --limit 0 --sleep-base 0.25 --sleep-jitter 0.10
#
# takes around 2 hrs

# %%
df = pl.read_parquet("output/curator_notes.parquet")
df

# %%
subset = df.filter(
    pl.col("curator_notes_html")
      .fill_null("")
      .str.contains("<i>")
)
subset

# %%
subset.filter(
    pl.col("gene_id") == "DDB_G0283907"
).select("curator_notes_html").to_series()[0]


# %% [markdown]
# ## summarize the year

# %%
year_pat = r"\b(18\d{2}|19\d{2}|20\d{2})\b"

year_counts = (
    subset
    .select(
        pl.col("curator_notes_html")
          .fill_null("")
          .str.extract_all(year_pat)   # -> list[str] of years per row
          .alias("years")
    )
    .explode("years")                 # one year per row
    .filter(pl.col("years").is_not_null() & (pl.col("years") != ""))
    .with_columns(pl.col("years").cast(pl.Int32).alias("year"))
    .group_by("year")
    .agg(pl.len().alias("n"))         # occurrences (mentions), not unique
    .sort("year")
)

year_counts

# %%
hits_2029 = (
    subset
    .filter(
        pl.col("curator_notes_html").fill_null("").str.contains(r"\b2029\b")
        # or use curator_notes_plain if you counted on plain
    )
    .select(["gene_id", "curator_notes_html", "curator_notes_plain"])
)

hits_2029[0,2]


# %% [markdown]
# ## let us quickly if two runs are consistent

# %%
df = pl.read_parquet("output/curator_notes.parquet")
df_ = pl.read_parquet("output/curator_notes.parquet_")
df.shape, df_.shape

# %%
cols = ["gene_id", "curator_notes_html", "curator_notes_plain"]

a = df.select(cols).rename({
    "curator_notes_html": "html_a",
    "curator_notes_plain": "plain_a",
})
b = df_.select(cols).rename({
    "curator_notes_html": "html_b",
    "curator_notes_plain": "plain_b",
})

diff = (
    a.join(b, on="gene_id", how="outer")
     .with_columns([
         # treat null == null as equal
         (pl.col("html_a").fill_null("")  != pl.col("html_b").fill_null("")).alias("html_diff"),
         (pl.col("plain_a").fill_null("") != pl.col("plain_b").fill_null("")).alias("plain_diff"),
     ])
     .filter(pl.col("html_diff") | pl.col("plain_diff"))
     .select([
         "gene_id",
         "html_diff", "plain_diff",
         "html_a", "html_b",
         "plain_a", "plain_b",
     ])
     .sort("gene_id")
)

# %%
with pl.Config(
    tbl_rows=diff.height,      # show all rows
    tbl_cols=20,               # or higher if needed
    fmt_str_lengths=100     # avoid string truncation
):
    print(diff)


# %% [markdown]
# this looks good, two fetch/runs do not differ much

# %% [markdown]
# ## claims and pulications

# %%
claims = pl.read_parquet("output/curator_claims.parquet")
claims

# %%
publications = pl.read_parquet("output/publications.parquet")
publications

# %%
(publications
    .filter(pl.col("year").is_not_null())
    .group_by("year")
    .agg(pl.len().alias("n_publications"))
    .sort("year")
)

# %% [markdown]
# ## check how much publication have pmids (overlapping with enl/xml file)

# %%
dicty21 = pl.read_csv("dictybase_files/dicty21/dicty21.xml.txt", separator="\t")
dicty21

# %%
pub = publications.select(
    pl.col("publication_id").cast(pl.Utf8).alias("publication_id"),
    pl.col("year").cast(pl.Int32, strict=False).alias("year"),
)

dicty = dicty21.select(
    pl.col("rec_number").cast(pl.Utf8).alias("rec_number")
)

# %%
dicty21.filter(
    pl.col("rec_number") == 247
)


# %% [markdown]
# we can see rec-number is not the internal publication id, sad!!

# %%

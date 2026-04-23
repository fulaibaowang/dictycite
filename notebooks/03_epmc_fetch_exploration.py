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
from turtle import pu
import requests
import time
import pandas as pd
import random
import html


# %% [markdown]
# # Download dicty literatures from EPMC

# %% [markdown]
# this contains interactive notebook from Jaka, in the end in the production I used article_fetch/ from Jakob

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## query EPMC 

# %%
BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


# %%
def search(query, cursor_mark):
    params = {
        'resultType': 'core',
        'format': 'json',
        'synonym': 'N',           # exact keyword
        'query': query,
        'pageSize': 1000,  
        'cursorMark': cursor_mark,
    }

    return requests.get(BASE_URL, params=params)


# %%
url_params = '''
(
   dictyostelium   
) 
AND (
    (HAS_FT:Y) AND 
    (SRC:MED OR SRC:PMC )
)
'''
pubmed_ids = []
cursor_mark = '*'

# %%
# more filters
# url_params = '''
# (
#    dictyostelium   
# ) 
# AND (
#     (FIRST_PDATE:[2000 TO 2025]) AND
#     (HAS_FT:Y) AND 
#     (((SRC:MED OR SRC:PMC OR SRC:AGR OR SRC:CBA) NOT (PUB_TYPE:"Review")))
# )
# '''

# %%
while True:
    response = search(url_params, cursor_mark)

    if len(pubmed_ids) == 1000: # <------------------------------remove this limit to get all results
        break 

    if response.status_code != 200:
        print(response.text)
        break
    data = response.json()
    print(data['hitCount'], len(data['resultList']['result']))

    if len(data['resultList']['result']) == 0:
        break

    for result in data['resultList']['result']: 
        if result['source'] in ('MED', 'PMC'):

            journal_info = result.get('journalInfo')
            journal = None
            if journal_info:
                # Medline abbreviation if available, otherwise journal title
                if 'medlineAbbreviation' in journal_info['journal']:
                    journal = journal_info['journal']['medlineAbbreviation']
                else:
                    journal = journal_info['journal']['title']


            pub_type = result['pubTypeList']['pubType']
            # if 'Abstract' in pub_type:
            #     continue

            # is_review = 'review-article' in pub_type or 'Review' in pub_type

            is_open_access = result['isOpenAccess'] == 'Y'
            cited_by_count = result['citedByCount']
            pub_year = result.get('pubYear')
            language = result.get('language')
            pub_type_list = ';'.join(result['pubTypeList']['pubType'])

            if 'keywordList' in result:
                keywords = ';'.join([keyword if keyword is not None else '' for keyword in result['keywordList']['keyword']])

            else:
                keywords = None

            # print(result['pmid'], result.get('pmcid'), pub_date, medline_abbreviation)
            pubmed_ids.append(
                (
                    # result.get('pmid'),
                    result.get('pmcid'),
                    int(is_open_access),
                    int(cited_by_count),
                    language,
                    pub_type_list,
                    pub_year,
                    journal,
                    result.get('title'),
                    keywords
                )
            )

    if 'nextCursorMark' not in data:
        break

    cursor_mark = data['nextCursorMark']

    time.sleep(random.uniform(0.5, 1.5))

# print(
#     f'Year: {year}, Number of articles saved: {len(pubmed_ids)} out of total {data["hitCount"]}'
# )

# %%
df = pd.DataFrame(
    pubmed_ids,
    columns=[
        # 'pmid',
        'pmcid',
        'is_open_access',
        'cited_by_count',
        'language',
        'pub_type_list',
        'pub_year',
        'journal',
        'title',
        'keywords',

    ],
)

# %%
df

# %%
summary = (df["pub_year"]
           .value_counts(dropna=False)
           .rename_axis("pub_year")
           .reset_index(name="count")
           .sort_values("pub_year"))

summary

# %%
df.to_csv(f'yun-dicty.csv', index=False)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## The below code will be used to get the BioC format for the articles and store them on disk.
#
# files will be saved as pmcid.json.gz

# %%
import requests
import time
import json 
import random
import os
import pandas as pd
import gzip


pmcids = list(df['pmcid'])[:5] # <------------------------------remove this limit to get all results


# # print(len(missing_pmcids - processed))

for pmcid in pmcids:
    url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode'
    response = requests.get(url)
    if response.status_code == 200:
        if 'No record can be found for the input:' in response.text or 'No result can be found' in response.text:
            print(f'No record for {pmcid}', flush=True)
            continue
        with gzip.open(f'{pmcid}.json.gz', 'wt') as f:
            f.write(json.dumps(response.json()[0], indent=2))
        print(f'{pmcid} saved', flush=True)
        # print(response.json())
    else:
        print(f'Error: {response.status_code}', flush=True)
    
    time.sleep(random.uniform(2, 5))


# %% [markdown]
# # literatures on dictybase

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## enl file
# load enl file from dictybase web page

# %%
from pathlib import Path
import zipfile
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter


# %% [markdown]
# But enl file seems not be able to parsed directly, I load enl file in ENDNOTE app and exported as xml file

# %%
xml_path = "dictybase_files/dicty21/dicty21.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

records = root.findall(".//record")
print("Total papers:", len(records))


# %%
rec = root.find(".//record")  # first record

def dump(elem, depth=0, max_depth=6):
    if depth > max_depth:
        return
    txt = (elem.text or "").strip()
    if txt:
        txt = txt[:120]
    print("  "*depth + f"<{elem.tag}> {txt}")
    for ch in list(elem):
        dump(ch, depth+1, max_depth=max_depth)

dump(rec)



# %%
pmid_re = re.compile(r"\bPMID[: ]*(\d{4,10})\b")

def style_texts(node):
    if node is None:
        return []
    styles = [s.text.strip() for s in node.findall(".//style") if s.text and s.text.strip()]
    if styles:
        return styles
    if node.text and node.text.strip():
        return [node.text.strip()]
    return []

def first_text_by_paths(rec, paths):
    """Try multiple XPath-like queries; return the first non-empty style/text."""
    for p in paths:
        node = rec.find(p)
        vals = style_texts(node)
        if vals:
            return vals[0]
    return ""

# candidate places where EndNote XML often stores abstracts
ABSTRACT_PATHS = [
    ".//abstract",                 # sometimes direct
    ".//abstracts/abstract",        # common
    ".//research-notes",            # sometimes used for abstract-ish text
    ".//notes",                     # fallback (may contain abstract or extra info)
]

rows = []

for event, rec in ET.iterparse(xml_path, events=("end",)):
    if rec.tag != "record":
        continue

    title = first_text_by_paths(rec, [".//titles/title"])
    year  = first_text_by_paths(rec, [".//dates/year"])
    rec_number = first_text_by_paths(rec, [".//rec-number", ".//key"])

    # authors (keep your current approach)
    author_nodes = rec.findall(".//contributors//author")
    authors = []
    for a in author_nodes:
        parts = style_texts(a)
        if parts:
            authors.append(parts[0])
    first_author = authors[0] if authors else ""

    # PMID (your logic)
    pmid = ""
    acc_txt = style_texts(rec.find(".//accession-num"))
    if acc_txt and acc_txt[0].isdigit():
        pmid = acc_txt[0]
    else:
        rec_str = ET.tostring(rec, encoding="unicode", method="xml")
        m = pmid_re.search(rec_str)
        if m:
            pmid = m.group(1)

    abstract = first_text_by_paths(rec, ABSTRACT_PATHS)

    rows.append({
        "rec_number": rec_number,
        "first_author": first_author,
        "year": year,
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
    })

    rec.clear()

df = pd.DataFrame(rows)
print("Total papers:", len(df))
print("With abstract:", df["abstract"].ne("").sum())


# %%
df

# %%
df.to_csv("dictybase_files/dicty21/dicty21.xml.txt", sep="\t", index=False)


# %%
# Year summary
df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
year_counts = (df.dropna(subset=["year_num"])
                 .assign(year_num=lambda x: x["year_num"].astype(int))
                 .value_counts("year_num")
                 .sort_index())

print(year_counts.tail(20))  # latest 20 years in the library

# PMID coverage
missing_pmid = df["pmid"].eq("").sum()
print(f"Missing PMID: {missing_pmid} / {len(df)} ({missing_pmid/len(df):.1%})")


# %%
df.loc[df["pmid"].ne(""), "pmid"].nunique()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## xls file (excel with gene name and literatures)

# %%
df_map = pd.read_csv("dictybase_files/DDBID_PMID.csv",sep=";")
pmids_ddbid = set(df_map["pubmed"].dropna().astype(int).astype(str))
print("Unique PMIDs in DDBID_PMID:", len(pmids_ddbid))

# %%
df_map

# %% [markdown]
# ### overlap between xml and csv(xls)

# %%
xml_pmids = (
    df.loc[df["pmid"].ne(""), "pmid"]
      .astype(str)
      .str.strip()
      .str.extract(r"(\d{4,10})", expand=False)   # keep just the digits
      .dropna()
)
xml_set = set(xml_pmids)

# %%
csv_pmids = (
    df_map["pubmed"]
      .dropna()
      .astype(str)
      .str.strip()
      .str.extract(r"(\d{4,10})", expand=False)
      .dropna()
)
csv_set = set(csv_pmids)

# %%
# --- 3) Overlap + differences ---
overlap = xml_set & csv_set
only_xml = xml_set - csv_set
only_csv = csv_set - xml_set

print("Overlap (unique PMIDs):", len(overlap))
print("Only in XML(df):", len(only_xml))
print("Only in CSV:", len(only_csv))

# optional: overlap percentages
print("Overlap as % of XML:", len(overlap)/len(xml_set) if xml_set else 0)
print("Overlap as % of CSV:", len(overlap)/len(csv_set) if csv_set else 0)

# %% [markdown]
# ## webpage

# %% [markdown]
# the above two files do not have all publications on dictybase, also it is diffcult to get internal publication ids there, so just ignore.
#
# it is possible to browser all pages of http://dictybase.org/publication/xxxx but there server is down often

# %% [markdown]
# e.g http://dictybase.org/publication/19729

# %% [markdown]
# In the end i used gene_publication_mapping.ipynb and scripts/public/data_prep/dicty_publication.py, works good

# %% [markdown]
# # check pmids in fetched from EPMC and dictybase

# %%
from pathlib import Path
import json
import polars as pl

# %% [markdown]
# ## jsons of EPMC

# %%
DATA_DIR = Path("scripts/public/article_fetching/output/all_cleaned")


# %%
rows = []
for fp in DATA_DIR.rglob("*.json"):
    try:
        with fp.open("r", encoding="utf-8") as f:
            d = json.load(f)

        abstract = d.get("abstract", None)
        text = d.get("text", None)

        # has_abstract: not null and not empty
        has_abstract = abstract is not None and str(abstract).strip() != ""

        # has_text: not null and contains some content
        if text is None:
            has_text = False
            n_text_sections = 0
            n_text_paras = 0
        elif isinstance(text, dict):
            n_text_sections = len(text)
            n_text_paras = sum(len(v) for v in text.values() if isinstance(v, list))
            has_text = (n_text_sections > 0) and (n_text_paras > 0)
        else:
            # in case text is stored as a string in some files
            has_text = str(text).strip() != ""
            n_text_sections = None
            n_text_paras = None

        rows.append({
            "pmid": d.get("pmid"),
            "pmcid": d.get("pmcid"),
            "doi": d.get("doi"),
            "year": d.get("year"),
            "title": d.get("title"),
            "journal": d.get("journal"),
            "authors": d.get("authors"),
            "has_abstract": has_abstract,
            "has_text": has_text,
            "n_text_sections": n_text_sections,
            "n_text_paras": n_text_paras,
            "file": str(fp),
        })

    except Exception as e:
        rows.append({
            "pmid": None,
            "pmcid": None,
            "doi": None,
            "year": None,
            "title": None,
            "journal": None,
            "authors": None,
            "has_abstract": False,
            "has_text": False,
            "n_text_sections": None,
            "n_text_paras": None,
            "file": str(fp),
            "error": str(e),
        })

df = pl.DataFrame(rows)

# %%
# your "pmid table"
pmid_table = (
    df.select(["pmid", "pmcid", "doi", "year", "title", "journal", "authors", "file"])
      .filter(pl.col("pmid").is_not_null())
)

# %%
pmid_table

# %% [markdown]
# ## check how this overlaps all publications on dictybase that has pmids

# %%
dicty_pmid_csv = "output/dicty_gold_build/2_publication_id_pmid.csv"
dicty_pmid = pl.read_csv(dicty_pmid_csv)

# %%
dicty_pmid


# %%
def normalize_pmid_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace(r"\.0$", "", literal=False)   # Excel float artifact
        .str.extract(r"(\d+)", 1)                 # keep digits only (first group)
    )

pmids_epmc = (
    pmid_table
    .select(pmid=normalize_pmid_expr("pmid"))
    .drop_nulls()
    .unique()
)

pmids_dictybase = (
    dicty_pmid
    .select(pmid=normalize_pmid_expr("pmid"))
    .drop_nulls()
    .unique()
)

# %%
overlap = pmids_epmc.join(pmids_dictybase, on="pmid", how="inner")
only_epmc  = pmids_epmc.join(pmids_dictybase, on="pmid", how="anti")
only_dictybase  = pmids_dictybase.join(pmids_epmc, on="pmid", how="anti")

# %%
summary = pl.DataFrame({
    "set": ["pmid_table", "dicty_pmid", "overlap", "only_epmc", "only_dictybase"],
    "unique_count": [pmids_epmc.height, pmids_dictybase.height, overlap.height, only_epmc.height, only_dictybase.height],
})

summary

# %% [markdown]
# it seems that many publication on dictybase were not fetched on epmc, let look at them

# %%
only_dictybase.head()

# %% [markdown]
# In the case, that the query is "Dictyostelium", we almost cover all the literature on dictybase, only 35 was missing. 
# This looks good!
#
# Earlier, when I query "OPEN_ACCESS:y AND \"Dictyostelium discoideum\", it covers only a bit more than 10%.
#
# We shall still look at all the literatues and see if they have abstract.
#

# %% [markdown]
# ## check in all literatures fetched, how many have abstract and text

# %%
# quick summary
summary = df.select([
    pl.len().alias("n_files"),
    pl.col("pmid").is_not_null().sum().alias("n_with_pmid"),
    pl.col("has_abstract").sum().alias("n_with_abstract"),
    pl.col("has_text").sum().alias("n_with_text"),
    (pl.col("has_abstract") & pl.col("has_text")).sum().alias("n_with_both"),
    (~pl.col("has_abstract") & ~pl.col("has_text")).sum().alias("n_with_neither"),
])

summary

# %% [markdown]
#  ## check in all literatures fetched that also on dictybase, how many have abstract and text

# %%
# 1) restrict df to overlap PMIDs
df_overlap = df.join(overlap, on="pmid", how="inner")

# 2) counts + fractions within overlap
overlap_summary = df_overlap.select([
    pl.len().alias("n_overlap_pmids"),
    pl.col("has_abstract").sum().alias("n_with_abstract"),
    pl.col("has_text").sum().alias("n_with_text"),
    (pl.col("has_abstract") & pl.col("has_text")).sum().alias("n_with_both"),
    (~pl.col("has_abstract") & ~pl.col("has_text")).sum().alias("n_with_neither"),
    (pl.col("has_abstract").mean()).alias("frac_with_abstract"),
    (pl.col("has_text").mean()).alias("frac_with_text"),
    ((pl.col("has_abstract") & pl.col("has_text")).mean()).alias("frac_with_both"),
])

overlap_summary


# %% [markdown]
# at least most of them have abstract!

# %% [markdown]
# ## let us save the table so we can use in datasets.ipynb

# %%
pmid_uniqueness = pmid_table.select([
    pl.col("pmid").is_not_null().sum().alias("n_nonnull_pmid"),
    pl.col("pmid").filter(pl.col("pmid").is_not_null()).n_unique().alias("n_unique_nonnull_pmid"),
]).with_columns(
    (pl.col("n_nonnull_pmid") - pl.col("n_unique_nonnull_pmid")).alias("n_duplicate_nonnull_pmid")
)

pmid_uniqueness

# %% [markdown]
# ok, no missing and duplicated pmids.

# %%
rows = []

for fp in DATA_DIR.rglob("*.json"):
    try:
        with fp.open("r", encoding="utf-8") as f:
            d = json.load(f)

        abstract = d.get("abstract", None)

        # normalize abstract to a string
        if abstract is None:
            continue
        if isinstance(abstract, list):
            abstract_str = "\n".join(str(x).strip() for x in abstract if str(x).strip() != "")
        else:
            abstract_str = str(abstract).strip()

        if abstract_str == "":
            continue  # only keep those that truly have abstracts

        text = d.get("text", None)

        # normalize text into two representations:
        # 1) nested text dict 그대로 (good for Parquet)
        text_nested = text if isinstance(text, dict) else None

        # 2) flattened plain text (useful for search / TSV)
        if text is None:
            text_plain = None
        elif isinstance(text, dict):
            parts = []
            for section, paras in text.items():
                if isinstance(paras, list):
                    # keep section header optionally
                    # parts.append(f"## {section}")
                    parts.extend([str(p).strip() for p in paras if str(p).strip() != ""])
            text_plain = "\n".join(parts).strip() if parts else None
        else:
            text_plain = str(text).strip() or None

        rows.append({
            "pmid": d.get("pmid"),
            "pmcid": d.get("pmcid"),
            "doi": d.get("doi"),
            "year": d.get("year"),
            "title": d.get("title"),
            "journal": d.get("journal"),
            "authors": d.get("authors"),
            "abstract": abstract_str,      # REAL abstract text
            "text_plain": text_plain,      # flattened full text (optional but handy)
            "text": text_nested,           # nested full text dict (optional; parquet only)
            "file": str(fp),
        })

    except Exception as e:
        # skip broken files (or keep them if you want)
        continue

df_abs = pl.DataFrame(rows)
df_abs = (
    df_abs.filter(pl.col("pmid").is_not_null())
)
df_abs

# %%
# (
#     df_abs
#     .drop(["text", "text_plain"])
#     .head(1000)
#     .write_csv("output/pmid_table_with_abstract.tsv", separator="\t")
# )


# %% [markdown]
# let us clean up HTML/JATS markup

# %%
def unescape(s: str | None) -> str | None:
    return None if s is None else html.unescape(s)

df_abs_clean = df_abs.with_columns(
    pl.col("abstract")
    # decode entities like &amp; &lt; etc
    .map_elements(unescape, return_dtype=pl.Utf8)
    # add newlines after headings / paragraphs / breaks
    .str.replace_all(r"</h[1-6]>", "\n")
    .str.replace_all(r"</p>", "\n")
    .str.replace_all(r"<br\s*/?>", "\n")
    # drop heading open tags (<h4 ...>) and all remaining tags (<i>, <sup>, <sub>, ...)
    .str.replace_all(r"<h[1-6][^>]*>", "")
    .str.replace_all(r"</?[^>]+>", "")
    # whitespace normalization
    .str.replace_all(r"\r\n", "\n")
    .str.replace_all(r"[ \t]+", " ")
    .str.replace_all(r"\n{3,}", "\n\n")
    .str.strip_chars()
    .alias("abstract_clean")
)

# %%
with_tags = df_abs.filter(pl.col("abstract").str.contains(r"<[^>]+>")).head(5)

with pl.Config(fmt_str_lengths=2000):
    display(
        df_abs_clean.join(with_tags.select("pmid"), on="pmid", how="inner")
                   .select(["pmid", "abstract", "abstract_clean"])
    )


# %%
df_abs_clean_small = df_abs_clean.select([
    "pmid",
    "pmcid",
    "doi",
    "year",
    "title",
    "journal",
    "authors",
    "abstract_clean",
    "file",
])

df_abs_clean_small.write_parquet("output/dicty_gold_build/3_articles_cleaned_abstract.parquet")

# %%

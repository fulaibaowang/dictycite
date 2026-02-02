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
import requests, json, re, time
import pandas as pd
from pathlib import Path
import json


# %%
key = Path("../llama_API_KEY").read_text().strip()
URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
MODEL = "llama3.3:latest"


# %% [markdown]
# # run llama

# %% [markdown]
# ## 1) Prompt template (your “good” prompt, parameterized)

# %%
PROMPT_TEMPLATE = """You are labeling curator-note **claim–citation** pairs for a retrieval benchmark in *Dictyostelium discoideum*.

### Input
You will receive:
- **CLAIM_QUERY**: a curator-note claim. It may already include entity expansion at the end (aliases/products).
- **PAPER_TITLE**
- **PAPER_ABSTRACT**

### Task
Return two labels:
1) whether this citation is a correct **document-level** match for the claim (for Phase-A style doc retrieval evaluation)
2) whether the **abstract alone** supports the claim’s core and/or detailed statement (for trust/evidence evaluation)

### Key concepts
**Core claim** = the main relationship/assertion (e.g., “loss of gene X causes chemotaxis defects”, “protein Y localizes to lysosomes”, “gene Z is required for development”).  
**Detail claim** = a specific sub-phenotype / qualifier / quantitative or mechanistic detail (e.g., “failure to suppress lateral pseudopods”, “high expression throughout growth and development”, “forms a separate phylogenetic clade”).

**Document-level match:** the paper is clearly the right citation for the claim’s **core claim**, even if the abstract does not contain every **detail claim**.  
**Abstract evidence:** the abstract must explicitly state the relevant relationship; do not infer unstated facts.

### Labels
Output these two fields:

**`doc_match`**
- `"yes"`: paper is clearly on-topic and supports the core claim at document level
- `"no"`: paper is clearly unrelated / wrong topic / wrong entity
- `"unclear"`: insufficient information or too ambiguous to judge reliably

**`evidence_level`**
- `"abstract_supports_detail"`: abstract explicitly supports the core claim **and** the key detail(s) stated in the claim
- `"abstract_supports_core"`: abstract explicitly supports the core claim, but not the key detail(s)
- `"needs_fulltext"`: paper seems plausibly on-topic, but the abstract does **not** explicitly support the core claim (or is too vague); full text likely needed
- `"not_applicable"`: only if `doc_match` is `"no"`

### Decision procedure (follow strictly)
1) Read the claim and identify:
   - main entity/entities and process (use the expanded aliases in the claim if present)
   - the **core claim**
   - any **detail claim(s)**

2) Scan title+abstract for explicit statements:
   - entity/topic alignment with the claim
   - explicit support for the **core claim**
   - explicit support for any **detail claim(s)**

3) Assign `doc_match`:
   - If title+abstract is clearly about a different entity/topic/process → `"no"`
   - If it is clearly about the same entity/topic and aligns with the claim’s core relationship → `"yes"`
   - If mixed/uncertain (claim vague; abstract vague; multiple entities) → `"unclear"`

4) Assign `evidence_level`:
   - If `doc_match="no"` → `"not_applicable"`
   - Else if abstract explicitly states the **detail claim(s)** (and core) → `"abstract_supports_detail"`
   - Else if abstract explicitly states the **core claim** but not the detail(s) → `"abstract_supports_core"`
   - Else → `"needs_fulltext"`

### Output format (JSON only, minimal)
Return ONLY this JSON object, with no extra text and no extra keys:

```json
{{
  "doc_match": "yes|no|unclear",
  "evidence_level": "abstract_supports_detail|abstract_supports_core|needs_fulltext|not_applicable",
  "reason": "max 25 words, concrete."
}}
```
Input begins

CLAIM_QUERY:
{claim_query}

PAPER_TITLE:
{title}

PAPER_ABSTRACT:
{abstract}

Input ends

"""

# %% [markdown]
# ## 2) API call function (with robust JSON extraction)
#
# LLMs sometimes wrap JSON in extra text. This parser grabs the first JSON object it sees.

# %%
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def call_llm(prompt: str, timeout=120) -> str:
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "stream": False, "prompt": prompt},
        timeout=timeout,
    )
    r.raise_for_status()
    data = json.loads(r.text)
    return data["response"]

def parse_label_json(text: str) -> dict:
    m = JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in response:\n{text[:500]}")
    return json.loads(m.group(0))


# %% [markdown]
# ## 3) Load TSV + run a few examples

# %%
TSV_PATH = "../output/cleaned/gold_with_query_expand_flat.tsv"  # <-- change
df = pd.read_csv(TSV_PATH, sep="\t", dtype=str).fillna("")
df.head()


# %%
sample = df.sample(n=3, random_state=0)  # or df.head(3)
sample[["group_claim_id", "pmid", "query_expand", "title", "abstract_clean"]]


# %%
# Test specific group_claim_id cases
test_ids = ["5", "15", "1262","1002","109"]
sample = df[df["group_claim_id"].isin(test_ids)].copy()
sample[["group_claim_id", "pmid", "query_expand", "title", "abstract_clean"]]

# %%
results = []

for _, row in sample.iterrows():
    group_claim_id = row["group_claim_id"]
    pmid = row["pmid"]
    claim_query = row["query_expand"] or row["query"]  # fallback if needed
    title = row["title"]
    abstract = row["abstract_clean"]

    prompt = PROMPT_TEMPLATE.format(
        claim_query=claim_query.strip(),
        title=title.strip(),
        abstract=abstract.strip(),
    )

    raw = call_llm(prompt)
    try:
        out = parse_label_json(raw)
    except Exception as e:
        out = {"doc_match": "unclear", "evidence_level": "needs_fulltext", "reason": f"parse_error: {e}"}

    # add your required keys
    out["group_claim_id"] = group_claim_id
    out["pmid"] = pmid

    results.append(out)

pd.DataFrame(results)


# %% [markdown]
# # production script: dicty_claim_labeler.py

# %% [markdown]
# we do this three times and check agreements across three replicates
#
# results:
#
# output/llm_labels_goldset_run1.jsonl
#
# output/llm_labels_goldset_run2.jsonl
#
# output/llm_labels_goldset_run3.jsonl

# %% [markdown]
# # Compare labels across three runs

# %%
def load_jsonl(path):
    rec = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec.append(json.loads(line))
    df = pd.DataFrame(rec)
    df["group_claim_id"] = df["group_claim_id"].astype(str)
    df["pmid"] = df["pmid"].astype(str)
    return df

run1 = load_jsonl("../output/llm_labels_goldset_run1.jsonl")
run2 = load_jsonl("../output/llm_labels_goldset_run2.jsonl")
run3 = load_jsonl("../output/llm_labels_goldset_run3.jsonl")

print(f"Run1: {len(run1)} records")
print(f"Run2: {len(run2)} records")
print(f"Run3: {len(run3)} records")

# Check for error cases
print("\n" + "=" * 60)
print("ERROR CHECK:")
print("=" * 60)

for run_num, df_run in [(1, run1), (2, run2), (3, run3)]:
    error_mask = df_run["reason"].str.contains("error:", na=False)
    n_errors = error_mask.sum()
    if n_errors > 0:
        print(f"\n⚠️  Run{run_num}: {n_errors} error cases found!")
        print(f"Error types in Run{run_num}:")
        print(df_run[error_mask]["reason"].value_counts())
    else:
        print(f"✓ Run{run_num}: No errors")

# Merge all three runs on (group_claim_id, pmid)
merged = (
    run1.rename(columns={
        "doc_match": "doc_match_1",
        "evidence_level": "evidence_1",
        "reason": "reason_1"
    })
    .merge(
        run2.rename(columns={
            "doc_match": "doc_match_2",
            "evidence_level": "evidence_2",
            "reason": "reason_2"
        }),
        on=["group_claim_id", "pmid"],
        how="outer"
    )
    .merge(
        run3.rename(columns={
            "doc_match": "doc_match_3",
            "evidence_level": "evidence_3",
            "reason": "reason_3"
        }),
        on=["group_claim_id", "pmid"],
        how="outer"
    )
)

print(f"Total pairs: {len(merged)}")
merged.head()


# %%
# Identify disagreements with custom logic
#
# For doc_match:
#   - Disagreement = when "yes" vs "no" OR "yes" vs "unclear"
#   - NO disagreement = when "no" vs "unclear"
#   (i.e., only "yes" paired with non-yes counts as disagreement)
#
# For evidence_level:
#   - Disagreement = when "abstract_supports_detail" or "abstract_supports_core" vs "needs_fulltext"
#   - NO disagreement = when "abstract_supports_detail" vs "abstract_supports_core"
#   (i.e., any explicit support vs needs_fulltext counts as disagreement)

def doc_match_disagree(v1, v2):
    """Returns True if v1 and v2 represent a meaningful disagreement."""
    pair = {v1, v2}
    # Disagreement only if "yes" is paired with "no" or "unclear"
    if "yes" in pair and ("no" in pair or "unclear" in pair):
        return True
    return False

def evidence_disagree(v1, v2):
    """Returns True if v1 and v2 represent a meaningful disagreement."""
    supports = {"abstract_supports_detail", "abstract_supports_core"}
    pair = {v1, v2}
    # Disagreement if one is explicit support and the other is needs_fulltext
    if (pair & supports) and "needs_fulltext" in pair:
        return True
    return False

# Check all pairwise comparisons for doc_match
merged["doc_match_disagree"] = (
    merged.apply(lambda r: 
        doc_match_disagree(r["doc_match_1"], r["doc_match_2"]) or
        doc_match_disagree(r["doc_match_1"], r["doc_match_3"]) or
        doc_match_disagree(r["doc_match_2"], r["doc_match_3"]),
        axis=1
    )
)

# Check all pairwise comparisons for evidence_level
merged["evidence_disagree"] = (
    merged.apply(lambda r: 
        evidence_disagree(r["evidence_1"], r["evidence_2"]) or
        evidence_disagree(r["evidence_1"], r["evidence_3"]) or
        evidence_disagree(r["evidence_2"], r["evidence_3"]),
        axis=1
    )
)

merged["any_disagree"] = merged["doc_match_disagree"] | merged["evidence_disagree"]

# Summary statistics
print("=" * 60)
print("Disagreement Statistics (custom logic):")
print("=" * 60)
print(f"Total pairs: {len(merged)}")
print(f"doc_match disagreements: {merged['doc_match_disagree'].sum()} ({100*merged['doc_match_disagree'].mean():.1f}%)")
print(f"evidence_level disagreements: {merged['evidence_disagree'].sum()} ({100*merged['evidence_disagree'].mean():.1f}%)")
print(f"Any disagreement: {merged['any_disagree'].sum()} ({100*merged['any_disagree'].mean():.1f}%)")
print(f"Full agreement: {(~merged['any_disagree']).sum()} ({100*(~merged['any_disagree']).mean():.1f}%)")

# %%
# Show examples of disagreements
disagreements = merged[merged["any_disagree"]].copy()

# Join with original data to see the query and abstract
disagreements_full = disagreements.merge(
    df[["group_claim_id", "pmid", "query_expand", "query", "title", "abstract_clean"]],
    on=["group_claim_id", "pmid"],
    how="left"
)

print(f"\nShowing {min(3, len(disagreements_full))} examples with disagreements:\n")

for i, row in disagreements_full.head(5).iterrows():
    print("=" * 80)
    print(f"Group Claim ID: {row['group_claim_id']} | PMID: {row['pmid']}")
    print("-" * 80)
    print(f"QUERY: {row['query_expand'] or row['query']}")
    print(f"\nTITLE: {row['title']}")
    print(f"\nABSTRACT: {row['abstract_clean'][:300]}...")
    print("\n" + "-" * 80)
    print("LABELS:")
    print(f"  Run1: doc_match={row['doc_match_1']:20s} | evidence={row['evidence_1']}")
    print(f"  Run2: doc_match={row['doc_match_2']:20s} | evidence={row['evidence_2']}")
    print(f"  Run3: doc_match={row['doc_match_3']:20s} | evidence={row['evidence_3']}")
    print("\nREASONS:")
    print(f"  Run1: {row['reason_1']}")
    print(f"  Run2: {row['reason_2']}")
    print(f"  Run3: {row['reason_3']}")
    print("=" * 80)
    print()

# %% [markdown]
# **Full agreement: 2028 (95.7%)**, this is good

# %% [markdown]
# # Summary: Agreement Tables

# %%

# --- Agreement flags using your custom logic ---
supports = {"abstract_supports_detail", "abstract_supports_core"}

has_yes = merged[["doc_match_1","doc_match_2","doc_match_3"]].eq("yes").any(axis=1)
has_no_or_unclear = merged[["doc_match_1","doc_match_2","doc_match_3"]].isin(["no","unclear"]).any(axis=1)
doc_match_agreement = (~(has_yes & has_no_or_unclear)).astype(int)

has_support = merged[["evidence_1","evidence_2","evidence_3"]].isin(supports).any(axis=1)
has_needs = merged[["evidence_1","evidence_2","evidence_3"]].eq("needs_fulltext").any(axis=1)
evidence_agreement = (~(has_support & has_needs)).astype(int)

# --- Table 1: agreement summary ---
agreement_table = merged[["group_claim_id","pmid"]].copy()
agreement_table["doc_match_agreement"] = doc_match_agreement
agreement_table["evidence_level_agreement"] = evidence_agreement


# %%
agreement_table.head()

# %%
# --- Table 2: fully agreed subset (both dimensions) ---
mask_full = (doc_match_agreement.astype(bool) & evidence_agreement.astype(bool))

full_agreement_table = merged.loc[mask_full, ["group_claim_id","pmid"]].copy()

# pick a representative label for doc_match:
# if any yes -> yes, else if any no -> no, else unclear
full_agreement_table["doc_match"] = (
    merged.loc[mask_full, ["doc_match_1","doc_match_2","doc_match_3"]]
    .apply(lambda r: "yes" if (r=="yes").any() else ("no" if (r=="no").any() else "unclear"), axis=1)
)

# pick a representative evidence label:
# if any supports_detail -> detail, elif any supports_core -> core, else needs_fulltext
full_agreement_table["evidence_level"] = (
    merged.loc[mask_full, ["evidence_1","evidence_2","evidence_3"]]
    .apply(lambda r: "abstract_supports_detail" if (r=="abstract_supports_detail").any()
                   else ("abstract_supports_core" if (r=="abstract_supports_core").any()
                         else "needs_fulltext"),
           axis=1)
)

# %%
full_agreement_table.head()

# %%
# Stats
print("Total pairs:", len(agreement_table))
print("Doc agreement:", agreement_table["doc_match_agreement"].mean())
print("Evidence agreement:", agreement_table["evidence_level_agreement"].mean())
print("Fully agreed:", len(full_agreement_table))

print("\ndoc_match stats (fully agreed):")
print(full_agreement_table["doc_match"].value_counts(dropna=False))

print("\nevidence_level stats (fully agreed):")
print(full_agreement_table["evidence_level"].value_counts(dropna=False))


# %%
# Save
out_dir = Path("../output")
out_dir.mkdir(parents=True, exist_ok=True)

agreement_table.to_csv(out_dir / "llama_agreement_summary.tsv", sep="\t", index=False)
full_agreement_table.to_csv(out_dir / "llama_full_agreement_cases.tsv", sep="\t", index=False)

print("Saved tables to", out_dir)

# %% [markdown]
# we will use
# **doc_match=yes, evidence=abstract_supports_core  | abstract_supports_detail** as gold sets

# %% [markdown]
# we can intesect **output/cleaned/gold_with_query_expand.parquet** and **output/cleaned/gold_with_query_expand_flat.tsv** to get the filter dataset

# %% [markdown]
# TODO: current expansion adds some noises, e.g AT, and might needs fix

# %%

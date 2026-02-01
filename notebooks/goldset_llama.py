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
# # 1) Prompt template (your “good” prompt, parameterized)

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
# # 3) Load TSV + run a few examples

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
# # production script is dicty_claim_labeler.py

# %%

# %% [markdown]
# ## Compare labels across three runs

# %%
def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

run1 = load_jsonl("../output/llm_labels_goldset_run1.jsonl")
run2 = load_jsonl("../output/llm_labels_goldset_run2.jsonl")
run3 = load_jsonl("../output/llm_labels_goldset_run3.jsonl")

print(f"Run1: {len(run1)} records")
print(f"Run2: {len(run2)} records")
print(f"Run3: {len(run3)} records")

# %%
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
disagreements = merged[~merged["all_same"]].copy()

# Join with original data to see the query and abstract
disagreements_full = disagreements.merge(
    df[["group_claim_id", "pmid", "query_expand", "query", "title", "abstract_clean"]],
    on=["group_claim_id", "pmid"],
    how="left"
)

print(f"\nShowing {min(5, len(disagreements_full))} examples with disagreements:\n")

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

# %%
# Breakdown of disagreement types
print("Disagreement breakdown:\n")

doc_disagree = merged[~merged["doc_match_same"]]
print(f"doc_match disagreements: {len(doc_disagree)}")
if len(doc_disagree) > 0:
    print("  Examples of doc_match variations:")
    for _, row in doc_disagree.head(3).iterrows():
        print(f"    Group {row['group_claim_id']}, PMID {row['pmid']}: {row['doc_match_1']} / {row['doc_match_2']} / {row['doc_match_3']}")

print()

evidence_disagree = merged[~merged["evidence_same"]]
print(f"evidence_level disagreements: {len(evidence_disagree)}")
if len(evidence_disagree) > 0:
    print("  Examples of evidence_level variations:")
    for _, row in evidence_disagree.head(3).iterrows():
        print(f"    Group {row['group_claim_id']}, PMID {row['pmid']}:")
        print(f"      {row['evidence_1'][:30]:30s} / {row['evidence_2'][:30]:30s} / {row['evidence_3'][:30]:30s}")

# %%

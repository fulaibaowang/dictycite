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


# %%
key = Path("../llama_API_KEY").read_text().strip()
URL = "https://chat.fri.uni-lj.si/ollama/api/generate"
MODEL = "llama3.3:latest"


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

"""

# %%

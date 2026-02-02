#!/usr/bin/env python3
"""
Label curator-note claim–citation pairs using an Ollama-compatible /api/generate endpoint.

Input: TSV with columns:
  group_claim_id, query, query_expand, title, abstract_clean, pmid

Output: JSONL with one record per (group_claim_id, pmid):
  {"group_claim_id": "...", "pmid": "...", "doc_match": "...", "evidence_level": "...", "reason": "..."}

Features:
- Resume-safe: skips keys already present in output JSONL
- Retries with backoff
- Optional abstract truncation
- Polite rate limiting
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Set

import pandas as pd
import requests


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
**Core claim** = the main relationship/assertion.
**Detail claim** = a specific sub-phenotype / qualifier / quantitative or mechanistic detail.

**Document-level match:** the paper is clearly the right citation for the claim’s **core claim**, even if the abstract does not contain every **detail claim**.
**Abstract evidence:** the abstract must explicitly state the relevant relationship; do not infer unstated facts.

### Labels
**doc_match**: "yes" | "no" | "unclear"
**evidence_level**: "abstract_supports_detail" | "abstract_supports_core" | "needs_fulltext" | "not_applicable"

### Decision procedure (follow strictly)
1) Identify core vs detail in the claim
2) Check title+abstract for explicit support
3) Assign doc_match
4) Assign evidence_level

### Output format (JSON only, minimal)
Return ONLY this JSON object, with no extra text and no extra keys:

```json
{{
  "doc_match": "yes|no|unclear",
  "evidence_level": "abstract_supports_detail|abstract_supports_core|needs_fulltext|not_applicable",
  "reason": "max 25 words, concrete."
}}
```

### Input begins
CLAIM_QUERY:
{claim_query}

PAPER_TITLE:
{title}

PAPER_ABSTRACT:
{abstract}
### Input ends
"""

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_DOC_MATCH = {"yes", "no", "unclear"}
VALID_EVIDENCE = {
    "abstract_supports_detail",
    "abstract_supports_core",
    "needs_fulltext",
    "not_applicable",
}


def load_key(key_path: str) -> str:
    return Path(key_path).read_text(encoding="utf-8").strip()


def load_done_keys(jsonl_path: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                gid = str(obj.get("group_claim_id", "")).strip()
                pmid = str(obj.get("pmid", "")).strip()
                if gid and pmid:
                    done.add((gid, pmid))
            except Exception:
                # ignore malformed lines
                continue
    return done


def parse_label_json(text: str) -> Dict[str, str]:
    """
    Extract the first JSON object from LLM response and validate fields.
    """
    m = JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in model response.")
    obj = json.loads(m.group(0))

    dm = obj.get("doc_match", "unclear")
    ev = obj.get("evidence_level", "needs_fulltext")
    rs = obj.get("reason", "")

    dm = dm if isinstance(dm, str) else str(dm)
    ev = ev if isinstance(ev, str) else str(ev)
    rs = rs if isinstance(rs, str) else str(rs)

    dm = dm.strip().lower()
    ev = ev.strip()

    if dm not in VALID_DOC_MATCH:
        dm = "unclear"
    if ev not in VALID_EVIDENCE:
        ev = "needs_fulltext"

    # keep reason short
    rs = " ".join(rs.strip().split())
    if len(rs) > 200:
        rs = rs[:200]

    return {"doc_match": dm, "evidence_level": ev, "reason": rs}


def call_llm(
    session: requests.Session,
    url: str,
    model: str,
    bearer_key: str,
    prompt: str,
    timeout_s: int,
) -> str:
    r = session.post(
        url,
        headers={"Authorization": f"Bearer {bearer_key}"},
        json={"model": model, "stream": False, "prompt": prompt},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = json.loads(r.text)
    return data["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", required=True, help="Input TSV path")
    ap.add_argument("--output_jsonl", required=True, help="Output JSONL path (append/resume)")
    ap.add_argument("--key_path", default="../llama_API_KEY", help="Path to bearer token file")
    ap.add_argument("--url", default="https://chat.fri.uni-lj.si/ollama/api/generate", help="API endpoint")
    ap.add_argument("--model", default="llama3.3:latest", help="Model name")
    ap.add_argument("--sleep_s", type=float, default=0.05, help="Sleep between requests")
    ap.add_argument("--timeout_s", type=int, default=120, help="Request timeout in seconds")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retries per row")
    ap.add_argument("--abstract_max_chars", type=int, default=4500, help="Truncate abstract to this many chars")
    ap.add_argument("--progress_every", type=int, default=50, help="Print progress every N new requests")
    ap.add_argument("--raise_on_error", type=bool, default=True, help="Raise exception on errors (default: True). Set False to silently continue.")
    args = ap.parse_args()

    bearer_key = load_key(args.key_path)

    df = pd.read_csv(args.input_tsv, sep="\t", dtype=str).fillna("")

    required = ["group_claim_id", "pmid", "query", "query_expand", "title", "abstract_clean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in TSV: {missing}")

    done = load_done_keys(args.output_jsonl)
    print(f"Loaded rows: {len(df)}")
    print(f"Already labeled (resume): {len(done)}")

    session = requests.Session()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    sent = 0
    skipped = 0

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            gid = str(row.group_claim_id).strip()
            pmid = str(row.pmid).strip()
            key2 = (gid, pmid)

            if key2 in done:
                skipped += 1
                continue

            claim = (row.query_expand or row.query or "").strip()
            title = (row.title or "").strip()
            abstract = (row.abstract_clean or "").strip()

            if args.abstract_max_chars and len(abstract) > args.abstract_max_chars:
                abstract = abstract[: args.abstract_max_chars] + "..."

            prompt = PROMPT_TEMPLATE.format(
                claim_query=claim,
                title=title,
                abstract=abstract,
            )

            last_err = None
            parsed = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    raw = call_llm(
                        session=session,
                        url=args.url,
                        model=args.model,
                        bearer_key=bearer_key,
                        prompt=prompt,
                        timeout_s=args.timeout_s,
                    )
                    parsed = parse_label_json(raw)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * attempt)

            if last_err is not None or parsed is None:
                error_msg = f"error:{type(last_err).__name__}" if last_err else "error:unknown"
                if args.raise_on_error:
                    raise RuntimeError(f"[Group {gid}, PMID {pmid}] Failed after {args.max_retries} retries: {last_err}")
                # Silent fallback (only if raise_on_error=False)
                parsed = {
                    "doc_match": "unclear",
                    "evidence_level": "needs_fulltext",
                    "reason": error_msg,
                }

            record = {
                "group_claim_id": gid,
                "pmid": pmid,
                "doc_match": parsed["doc_match"],
                "evidence_level": parsed["evidence_level"],
                "reason": parsed["reason"],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            done.add(key2)
            sent += 1

            if args.progress_every and sent % args.progress_every == 0:
                print(f"[row {i}/{len(df)}] new={sent} skipped={skipped} total_done={len(done)}")

            if args.sleep_s:
                time.sleep(args.sleep_s)

    print("Done.")
    print(f"New labeled: {sent}")
    print(f"Skipped existing: {skipped}")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble the unified retrieval corpus.

Merges curated abstracts and (optionally) PDF body/caption chunks into a single
JSONL corpus, with chunk-level docnos so retrieval and eval can distinguish
chunks of the same paper.

Usage:
    # abstracts-only (rebuild of the legacy corpus with new docno scheme)
    python -m scripts.public.pdf_processing.build_corpus \\
        --abstracts dicty_simulated_data/abstracts/corpus.jsonl \\
        --out dicty_simulated_data/abstracts_v2/corpus.jsonl

    # abstracts + chunks (new full-text corpus)
    python -m scripts.public.pdf_processing.build_corpus \\
        --abstracts dicty_simulated_data/abstracts/corpus.jsonl \\
        --chunks output/pdf_extraction/v1/chunks.jsonl \\
        --out dicty_fulltext_corpus/corpus.jsonl

Output schema (one JSON record per line):
    {
      "docno": "<pmid>#abstract" | "<pmid>#body_001" | "<pmid>#caption_001",
      "pmid":  "<pmid>",
      "title": "<title>",
      "type":  "abstract" | "body" | "caption",
      "text":  "<text>",
      "seq":   1,                # chunks only
      "n_chars": 1842,           # chunks only
      "position_frac": 0.0       # body chunks only
    }

The `docno` is unique across the whole corpus; `pmid` groups chunks belonging to
the same paper. Eval-time max-pool aggregation uses `pmid` as the group key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_titles(path: Path) -> dict[str, str]:
    """Load a {pmid: title} map.

    Accepts two formats, auto-detected:
      - JSON object: {"<pmid>": "<title>", ...}  (e.g. titles.json from fetch_titles.py)
      - JSONL with one record per line, each having a 'pmid' (or 'PMID') and 'title' field
        (e.g. output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl)
    """
    text = path.read_text()
    head = text.lstrip()
    if head.startswith("{") and not head.startswith("{\""):
        data = json.loads(text)
        return {str(k): str(v or "") for k, v in data.items() if v}
    if head.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and not any(isinstance(v, dict) for v in data.values()):
                return {str(k): str(v or "") for k, v in data.items() if v}
        except json.JSONDecodeError:
            pass
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pmid = str(r.get("pmid") or r.get("PMID") or "").strip()
        title = (r.get("title") or "").strip()
        if pmid and title:
            out[pmid] = title
    return out


def iter_abstracts(path: Path, titles: dict[str, str] | None = None) -> list[dict]:
    """Read an abstracts corpus and emit unified-schema rows.

    Accepts either of two input row schemas (auto-detected per row):
      - legacy abstracts corpus: {"pmid", "title", "abstract"}
      - 7c_articles_cleaned_abstract.jsonl (and similar):
        {"pmid", "title", "text", "type": "abstract", ...}

    Output docno: <pmid>#abstract. Titles are taken from the optional titles map
    when present; otherwise fall back to the row's own title field.
    """
    titles = titles or {}
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            pmid = str(r.get("pmid", "")).strip()
            if not pmid:
                continue
            text = (r.get("abstract") or r.get("text") or "").strip()
            title = titles.get(pmid) or r.get("title") or ""
            out.append({
                "docno": f"{pmid}#abstract",
                "pmid": pmid,
                "title": title,
                "type": "abstract",
                "text": text,
            })
    return out


def iter_chunks(path: Path, titles: dict[str, str] | None = None) -> list[dict]:
    """Read the chunked PDF corpus and emit unified-schema rows.

    Input rows already carry chunk_id like "<pmid>#body_001"; we promote that to
    docno and pass through the remaining fields. Titles are taken from the
    optional titles map when present; otherwise fall back to the row's own
    title field (which chunk_bodies.py already populates from titles.json).
    """
    titles = titles or {}
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            chunk_id = str(r.get("chunk_id") or "").strip()
            pmid = str(r.get("pmid") or "").strip()
            if not chunk_id or not pmid:
                continue
            row = {
                "docno": chunk_id,
                "pmid": pmid,
                "title": titles.get(pmid) or r.get("title") or "",
                "type": r.get("type") or "body",
                "text": r.get("text") or "",
            }
            for k in ("seq", "n_chars", "position_frac"):
                if k in r:
                    row[k] = r[k]
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abstracts", type=Path, required=True,
                    help="Legacy abstracts corpus.jsonl (schema: pmid, title, abstract).")
    ap.add_argument("--chunks", type=Path, default=None,
                    help="Optional chunks.jsonl from the PDF pipeline. Omit for abstracts-only.")
    ap.add_argument("--titles", type=Path, action="append", default=[],
                    help="Optional title source: JSON {pmid: title} or JSONL with pmid+title fields. "
                         "Used to backfill/override titles for both abstract and chunk rows. "
                         "Repeatable; later sources win on duplicate PMIDs.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output corpus.jsonl path.")
    args = ap.parse_args()

    if not args.abstracts.exists():
        print(f"ERROR: abstracts file not found: {args.abstracts}", file=sys.stderr)
        return 1
    if args.chunks and not args.chunks.exists():
        print(f"ERROR: chunks file not found: {args.chunks}", file=sys.stderr)
        return 1

    titles_map: dict[str, str] = {}
    for tp in args.titles:
        if not tp.exists():
            print(f"ERROR: titles file not found: {tp}", file=sys.stderr)
            return 1
        loaded = load_titles(tp)
        titles_map.update(loaded)
        print(f"Loaded {len(loaded)} titles from {tp}.")
    if args.titles:
        print(f"Merged titles map: {len(titles_map)} unique PMIDs.")

    rows = iter_abstracts(args.abstracts, titles_map)
    n_abstracts = len(rows)
    n_chunks = 0
    if args.chunks:
        chunk_rows = iter_chunks(args.chunks, titles_map)
        rows.extend(chunk_rows)
        n_chunks = len(chunk_rows)

    seen_docnos = set()
    dupes: list[str] = []
    for r in rows:
        if r["docno"] in seen_docnos:
            dupes.append(r["docno"])
        seen_docnos.add(r["docno"])
    if dupes:
        print(f"ERROR: {len(dupes)} duplicate docnos (first 5: {dupes[:5]})", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pmids = len({r["pmid"] for r in rows})
    n_with_title = sum(1 for r in rows if r["title"])
    print(f"Wrote {len(rows)} docs to {args.out}")
    print(f"  abstracts: {n_abstracts}")
    print(f"  chunks:    {n_chunks}")
    print(f"  unique pmids: {n_pmids}")
    print(f"  rows with title: {n_with_title}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

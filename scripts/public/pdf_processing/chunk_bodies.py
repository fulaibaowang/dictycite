#!/usr/bin/env python3
"""Chunk cleaned body texts into chunks.jsonl ready for retrieval indexing.

Usage:
    python -m scripts.public.pdf_processing.chunk_bodies \\
        --in output/pdf_extraction/v1 \\
        --out output/pdf_extraction/v1/chunks.jsonl

Reads:
    <in>/body/<pmid>.txt
    <in>/flag_report.jsonl   (used to skip PMIDs that errored during cleaning)

Writes:
    <out>  one JSON per line, schema:
        {pmid, chunk_id, type ("body"|"caption"), seq, text, n_chars, position_frac}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunk import chunk_body, split_captions_from_body
from .config import CHUNK_CAP, CHUNK_MIN, CHUNK_OVERLAP, CHUNK_TARGET, DEFAULT_OUT_DIR

# tier4 flags that disqualify a PMID from chunking. low_chars_per_page is the
# scan signal (body < 500 chars/page) — chunking such bodies produces junk.
EXCLUDED_TIER4_FLAGS = frozenset({"low_chars_per_page"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Input dir containing body/ and flag_report.jsonl (default: {DEFAULT_OUT_DIR}).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSONL path (default: <in>/chunks.jsonl).")
    ap.add_argument("--titles", type=Path, default=None,
                    help="Optional {pmid: title} JSON; when given, each chunk row carries the paper title. "
                         "Defaults to <in>/titles.json if it exists.")
    ap.add_argument("--target", type=int, default=CHUNK_TARGET,
                    help=f"Target chunk size in chars (default: {CHUNK_TARGET}).")
    ap.add_argument("--cap", type=int, default=CHUNK_CAP,
                    help=f"Hard cap on a single chunk (default: {CHUNK_CAP}).")
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP,
                    help=f"Overlap between chunks in chars (default: {CHUNK_OVERLAP}).")
    ap.add_argument("--min", dest="min_size", type=int, default=CHUNK_MIN,
                    help=f"Min chunk size; tail < min merges into prev (default: {CHUNK_MIN}).")
    args = ap.parse_args()

    out_path = args.out or (args.in_dir / "chunks.jsonl")
    flag_path = args.in_dir / "flag_report.jsonl"
    body_dir = args.in_dir / "body"
    titles_path = args.titles or (args.in_dir / "titles.json")

    if not flag_path.exists():
        print(f"ERROR: missing {flag_path}", file=sys.stderr)
        return 1
    if not body_dir.is_dir():
        print(f"ERROR: missing {body_dir}", file=sys.stderr)
        return 1

    titles: dict[str, str] = {}
    if titles_path.exists():
        titles = json.loads(titles_path.read_text())
        print(f"Loaded {len(titles)} titles from {titles_path}.")
    else:
        print(f"NOTE: no titles file at {titles_path}; chunks will have empty title field.", file=sys.stderr)

    with flag_path.open() as f:
        cleaned_records = [json.loads(line) for line in f]

    rows: list[dict] = []
    n_skipped_error = n_skipped_flag = 0
    for rec in cleaned_records:
        if "error" in rec:
            n_skipped_error += 1
            continue
        if set(rec.get("tier4_flags") or []) & EXCLUDED_TIER4_FLAGS:
            n_skipped_flag += 1
            continue
        pmid = rec["pmid"]
        body_file = body_dir / f"{pmid}.txt"
        if not body_file.exists():
            continue
        body = body_file.read_text()
        body_no_caps, caps = split_captions_from_body(body)
        body_chunks = chunk_body(
            body_no_caps,
            target_size=args.target,
            hard_cap=args.cap,
            min_size=args.min_size,
            overlap=args.overlap,
        )
        body_total = len(body_no_caps) or 1

        title = titles.get(pmid, "")

        cum = 0
        for i, ck in enumerate(body_chunks, start=1):
            rows.append({
                "pmid": pmid,
                "title": title,
                "chunk_id": f"{pmid}#body_{i:03d}",
                "type": "body",
                "seq": i,
                "text": ck,
                "n_chars": len(ck),
                "position_frac": round(cum / body_total, 3),
            })
            cum += len(ck) - args.overlap

        for i, cap in enumerate(caps, start=1):
            rows.append({
                "pmid": pmid,
                "title": title,
                "chunk_id": f"{pmid}#caption_{i:03d}",
                "type": "caption",
                "seq": i,
                "text": cap,
                "n_chars": len(cap),
                "position_frac": None,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_pmids = len({r["pmid"] for r in rows})
    print(f"Wrote {len(rows)} chunks ({n_pmids} unique pmids) to {out_path}")
    if n_skipped_error or n_skipped_flag:
        print(f"Skipped: {n_skipped_error} extraction errors, {n_skipped_flag} tier4-excluded "
              f"(flags: {sorted(EXCLUDED_TIER4_FLAGS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

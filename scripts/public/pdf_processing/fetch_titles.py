#!/usr/bin/env python3
"""Fetch PubMed titles for a set of PMIDs and cache them as titles.json.

Usage:
    python -m scripts.public.pdf_processing.fetch_titles \\
        --pdfs-dir /Users/yun/Documents/dictybase_papers/pdfs \\
        --pdfs-dir /Users/yun/Documents/dictybase_papers/manual \\
        --out output/pdf_extraction/v1/titles.json

PMIDs are discovered by globbing *.pdf in each --pdfs-dir (filename = PMID).
Titles are fetched in batches via NCBI E-utilities esummary. If NCBI_API_KEY is
set in the environment, the request rate is bumped from 3 req/s to 10 req/s.

The output is a flat {pmid: title} JSON map. Re-runs merge with any existing
file: missing PMIDs are fetched, existing entries are kept (pass --overwrite to
force a full refetch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH_SIZE = 200
DELAY_NO_KEY = 0.34   # ~3 req/s
DELAY_WITH_KEY = 0.11 # ~10 req/s


def discover_pmids(pdf_dirs: list[Path]) -> set[str]:
    pmids: set[str] = set()
    for d in pdf_dirs:
        if not d.is_dir():
            print(f"WARNING: not a directory: {d}", file=sys.stderr)
            continue
        for p in d.glob("*.pdf"):
            stem = p.stem
            if stem.isdigit():
                pmids.add(stem)
    return pmids


def read_pmid_file(path: Path) -> set[str]:
    pmids: set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s.isdigit():
            pmids.add(s)
    return pmids


def fetch_batch(pmids: list[str], api_key: str | None) -> dict[str, str]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    url = f"{ESUMMARY_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "dictycite-pdf-processing/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out: dict[str, str] = {}
    result = data.get("result", {})
    for pmid in pmids:
        info = result.get(pmid)
        if isinstance(info, dict):
            title = (info.get("title") or "").strip()
            if title:
                out[pmid] = title
    return out


def fetch_titles(pmids: set[str], api_key: str | None) -> dict[str, str]:
    if not pmids:
        return {}
    delay = DELAY_WITH_KEY if api_key else DELAY_NO_KEY
    pmid_list = sorted(pmids)
    n_batches = (len(pmid_list) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Fetching titles for {len(pmid_list)} PMIDs in {n_batches} batch(es) "
          f"({'with' if api_key else 'without'} API key)...")
    titles: dict[str, str] = {}
    for i in range(0, len(pmid_list), BATCH_SIZE):
        batch = pmid_list[i : i + BATCH_SIZE]
        try:
            titles.update(fetch_batch(batch, api_key))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"  WARNING: batch starting at {batch[0]} failed: {exc}", file=sys.stderr)
        if i + BATCH_SIZE < len(pmid_list):
            time.sleep(delay)
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdfs-dir", type=Path, action="append", default=[],
                    help="Directory containing *.pdf files named by PMID (repeatable).")
    ap.add_argument("--pmids-file", type=Path, default=None,
                    help="File with one PMID per line (alternative to --pdfs-dir; both can be combined).")
    ap.add_argument("--out", type=Path, default=Path("output/pdf_extraction/v1/titles.json"),
                    help="Output JSON path (default: output/pdf_extraction/v1/titles.json).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Refetch all PMIDs even if already in the output file.")
    args = ap.parse_args()

    if not args.pdfs_dir and not args.pmids_file:
        ap.error("at least one of --pdfs-dir or --pmids-file is required")

    pmids = discover_pmids(args.pdfs_dir)
    if args.pmids_file:
        pmids |= read_pmid_file(args.pmids_file)
    print(f"Discovered {len(pmids)} PMIDs (dirs={len(args.pdfs_dir)}, pmids-file={'yes' if args.pmids_file else 'no'}).")

    existing: dict[str, str] = {}
    if args.out.exists() and not args.overwrite:
        existing = json.loads(args.out.read_text())
        print(f"Loaded {len(existing)} existing titles from {args.out}.")

    to_fetch = pmids if args.overwrite else (pmids - existing.keys())
    print(f"Need to fetch: {len(to_fetch)}")

    api_key = os.environ.get("NCBI_API_KEY")
    new_titles = fetch_titles(to_fetch, api_key)

    merged = {**existing, **new_titles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False))
    resolved = sum(1 for p in pmids if p in merged)
    print(f"Wrote {len(merged)} titles to {args.out} ({resolved}/{len(pmids)} of discovered PMIDs resolved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

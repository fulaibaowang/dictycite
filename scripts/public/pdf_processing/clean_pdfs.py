#!/usr/bin/env python3
"""Clean PDFs through Tier 1–4 passes; write body, references, and flag report.

Usage:
    python -m scripts.public.pdf_processing.clean_pdfs \\
        --pdfs-dir /Users/yun/Documents/dictybase_papers/pdfs \\
        --pdfs-dir /Users/yun/Documents/dictybase_papers/manual \\
        --titles output/pdf_extraction/v1/titles.json \\
        --pmids output/pdf_extraction/v1/target_pmids.txt \\
        --out output/pdf_extraction/v1

Behavior:
- PDFs are discovered by globbing *.pdf in each --pdfs-dir; filename stem = PMID.
- If --pmids is given, only those PMIDs are processed (one PMID per line).
- Later --pdfs-dir entries override earlier ones for the same PMID
  (e.g., put `manual/` after `pdfs/` to prefer manually-corrected copies).
- KNOWN_BAD PMIDs (see config.py) are skipped unless --include-known-bad is set.

Outputs (under --out):
    body/<pmid>.txt          cleaned body
    references/<pmid>.txt    reference list (when detected)
    flag_report.jsonl        per-PDF stats and Tier 3/4 flags
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .clean import split_structural, tier1_clean, tier3_layout_specific, tier4_validate
from .config import DEFAULT_OUT_DIR, KNOWN_BAD
from .extract import extract_pages_pymupdf


def discover_targets(pdf_dirs: list[Path], pmid_filter: set[str] | None) -> list[tuple[str, Path]]:
    """Return [(pmid, path)] with later dirs overriding earlier on PMID conflict.
    If pmid_filter is given, only PMIDs in the filter are kept."""
    by_pmid: dict[str, Path] = {}
    for d in pdf_dirs:
        if not d.is_dir():
            print(f"WARNING: not a directory: {d}", file=sys.stderr)
            continue
        for p in sorted(d.glob("*.pdf")):
            stem = p.stem
            if not stem.isdigit():
                continue
            if pmid_filter is not None and stem not in pmid_filter:
                continue
            by_pmid[stem] = p
    return sorted(by_pmid.items())


def process_one(pmid: str, pdf_path: Path, gold_title: str | None, out_dir: Path) -> dict:
    """Run the full pipeline on one PDF; return a stats row."""
    try:
        pages = extract_pages_pymupdf(pdf_path)
        if not pages:
            return {"pmid": pmid, "src": pdf_path.parent.name, "error": "no_pages"}
        pages_t3, t3_applied = tier3_layout_specific(pages)
        cleaned = tier1_clean(pages_t3)
        parts = split_structural(cleaned)
        flags = tier4_validate(
            body=parts["body"],
            n_pages=len(pages_t3),
            refs_found=bool(parts["references"]),
            gold_title=gold_title,
        )
        (out_dir / "body" / f"{pmid}.txt").write_text(parts["body"])
        if parts["references"]:
            (out_dir / "references" / f"{pmid}.txt").write_text(parts["references"])
        return {
            "pmid": pmid,
            "src": pdf_path.parent.name,
            "n_pages": len(pages),
            "n_pages_after_t3": len(pages_t3),
            "body_chars": len(parts["body"]),
            "refs_chars": len(parts["references"]),
            "ack_chars": len(parts["acknowledgments"]),
            "n_captions": len(parts["captions"]),
            "refs_found": bool(parts["references"]),
            "refs_via": parts["refs_via"],
            "tier3_applied": t3_applied,
            "tier4_flags": flags,
        }
    except Exception as e:
        return {"pmid": pmid, "src": pdf_path.parent.name, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdfs-dir", type=Path, action="append", required=True,
                    help="Directory of *.pdf files named by PMID (repeatable; later wins).")
    ap.add_argument("--titles", type=Path, required=True,
                    help="JSON map {pmid: title} produced by fetch_titles.py.")
    ap.add_argument("--pmids", type=Path, default=None,
                    help="Optional file with one PMID per line; restricts processing to these.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Output dir (default: {DEFAULT_OUT_DIR}).")
    ap.add_argument("--include-known-bad", action="store_true",
                    help="Process even PMIDs in config.KNOWN_BAD (default: skip).")
    args = ap.parse_args()

    titles: dict[str, str] = json.loads(args.titles.read_text())
    pmid_filter: set[str] | None = None
    if args.pmids:
        pmid_filter = {ln.strip() for ln in args.pmids.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        print(f"Restricting to {len(pmid_filter)} PMIDs from {args.pmids}.")

    targets = discover_targets(args.pdfs_dir, pmid_filter)
    if not args.include_known_bad:
        before = len(targets)
        targets = [(p, path) for p, path in targets if p not in KNOWN_BAD]
        skipped = before - len(targets)
        if skipped:
            print(f"Skipping {skipped} KNOWN_BAD PMIDs (use --include-known-bad to override).")
    print(f"Targets: {len(targets)} PDFs")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "body").mkdir(exist_ok=True)
    (args.out / "references").mkdir(exist_ok=True)

    t0 = time.time()
    rows: list[dict] = []
    for pmid, pdf_path in targets:
        rows.append(process_one(pmid, pdf_path, titles.get(pmid), args.out))
    elapsed = time.time() - t0
    if rows:
        print(f"Processed {len(rows)} PDFs in {elapsed:.1f}s ({elapsed/len(rows)*1000:.0f} ms/pdf avg)")

    flag_path = args.out / "flag_report.jsonl"
    with flag_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote flag report: {flag_path}")

    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  {r['pmid']} ({r.get('src','?')}): {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

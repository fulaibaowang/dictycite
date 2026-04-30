#!/usr/bin/env python3
# Convert the dicty articles JSONL (7c_articles_cleaned_abstract.jsonl) to
# Anserini's JsonCollection format for the ragnarok-style BM25 baseline.
#
# Anserini expects: {"id": "<docid>", "contents": "<text>"} per line.
# We mirror the project's PyTerrier BM25 input (title + "\n\n" + abstract)
# but DO NOT apply augment_text_for_codes() — keeping the public-baseline
# framing clean. See ragnarok_baseline plan in conversation history.
#
# Usage:
#   python convert_dicty_to_anserini.py \
#       --input  output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
#       --output_dir indexes/anserini_input/dicty \
#       --shard_size 50000

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def iter_anserini_docs(input_path: Path):
    skipped_no_pmid = 0
    skipped_empty = 0
    yielded = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            pmid = (d.get("pmid") or d.get("docno") or "").strip()
            if not pmid:
                skipped_no_pmid += 1
                continue

            title = (d.get("title") or "").strip()
            abstract = (d.get("abstract") or "").strip()
            text = (title + "\n\n" + abstract).strip()
            if not text:
                skipped_empty += 1
                continue

            yielded += 1
            yield {"id": pmid, "contents": text}

    print(
        f"[convert] yielded={yielded:,} skipped_no_pmid={skipped_no_pmid:,} "
        f"skipped_empty={skipped_empty:,}",
        file=sys.stderr,
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to 7c_articles_cleaned_abstract.jsonl")
    ap.add_argument("--output_dir", required=True, help="Directory to write Anserini JSONL shards into")
    ap.add_argument("--shard_size", type=int, default=50_000, help="Docs per shard file")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    in_shard = 0
    fh = None

    for rec in iter_anserini_docs(in_path):
        if fh is None or in_shard >= args.shard_size:
            if fh is not None:
                fh.close()
            shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
            fh = shard_path.open("w", encoding="utf-8")
            shard_idx += 1
            in_shard = 0
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        in_shard += 1

    if fh is not None:
        fh.close()

    print(f"[convert] wrote {shard_idx} shard(s) under {out_dir}")


if __name__ == "__main__":
    main()

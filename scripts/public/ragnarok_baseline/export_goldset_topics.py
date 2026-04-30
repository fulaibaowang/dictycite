#!/usr/bin/env python3
# Export the dicty 7a goldset to two artifacts the ragnarok-style baseline needs:
#   - topics.jsonl (Anserini -topicReader JsonString format: {"qid", "text"})
#   - qrels.txt    (TREC qrels: qid 0 docid rel)
#
# Query field defaults to "query_text" to match the project's BM25 stage
# (retrieve_bm25.py default). Relevance docids are pulled from docs[*].pmid.
#
# Usage:
#   python export_goldset_topics.py \
#       --input  output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl \
#       --topics_out  output/ragnarok_baseline/topics.jsonl \
#       --qrels_out   output/ragnarok_baseline/qrels.txt

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--topics_out", required=True)
    ap.add_argument("--qrels_out", required=True)
    ap.add_argument("--query_field", default="query_text")
    args = ap.parse_args()

    in_path = Path(args.input)
    topics_path = Path(args.topics_out)
    qrels_path = Path(args.qrels_out)
    topics_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    n_topics = 0
    n_qrels = 0
    n_skipped_no_text = 0
    n_skipped_no_docs = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         topics_path.open("w", encoding="utf-8") as ftopics, \
         qrels_path.open("w", encoding="utf-8") as fqrels:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            qid = str(d.get("query_id") or d.get("qid") or "").strip()
            text = (d.get(args.query_field) or "").strip()
            if not qid or not text:
                n_skipped_no_text += 1
                continue

            ftopics.write(json.dumps({"qid": qid, "text": text}, ensure_ascii=False) + "\n")
            n_topics += 1

            # Pull relevant docids from docs[*].pmid (preferred) or fall back to
            # parsing pubmed URLs in `documents`.
            rel_pmids = []
            for doc in d.get("docs") or []:
                pmid = str(doc.get("pmid") or "").strip()
                if pmid:
                    rel_pmids.append(pmid)
            if not rel_pmids:
                for url in d.get("documents") or []:
                    if "pubmed/" in url:
                        pmid = url.rsplit("/", 1)[-1].strip()
                        if pmid.isdigit():
                            rel_pmids.append(pmid)

            if not rel_pmids:
                n_skipped_no_docs += 1
                continue

            for pmid in dict.fromkeys(rel_pmids):  # de-dup, preserve order
                fqrels.write(f"{qid} 0 {pmid} 1\n")
                n_qrels += 1

    print(
        f"[export] topics={n_topics:,} qrels={n_qrels:,} "
        f"skipped_no_text={n_skipped_no_text:,} skipped_no_docs={n_skipped_no_docs:,}",
        file=sys.stderr, flush=True,
    )
    print(f"[export] wrote {topics_path}")
    print(f"[export] wrote {qrels_path}")


if __name__ == "__main__":
    main()

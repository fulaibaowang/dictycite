#!/usr/bin/env python3
# Score the ragnarok-baseline TREC runs with the project's own evaluator
# (retrieval_eval.common.evaluate_run), so the resulting metrics.csv has the
# same schema as the project's retrieval/{bm25,dense,fusion}/metrics.csv —
# direct row-by-row comparison.
#
# Inputs:
#   --qrels      qrels.txt (TREC: qid 0 docid rel)
#   --runs_dir   directory containing *_run.txt files (e.g. bm25_run.txt, rerank_run.txt)
# Output:
#   --output_csv metrics.csv with method,batch,n_queries,MAP@10,MRR@10,...

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Make retrieval_eval importable when run from the dictycite repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "public" / "shared_scripts"))

from retrieval_eval.common import RECALL_KS, BatchResult, evaluate_run  # noqa: E402


def parse_qrels(path: Path):
    gold = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                if int(rel) <= 0:
                    continue
            except ValueError:
                continue
            gold[str(qid)].append(str(docid))
    return dict(gold)


def parse_trec_run(path: Path):
    """TREC run: qid Q0 docid rank score run_id. Returns {qid: [docid,...]} in rank order."""
    by_q = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            by_q[str(qid)].append((int(rank), str(docid)))
    out = {}
    for qid, hits in by_q.items():
        hits.sort(key=lambda x: x[0])
        out[qid] = [d for _, d in hits]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--batch", default="7a_public_goldset",
                    help="Batch label written to metrics.csv (mirrors project pipeline)")
    args = ap.parse_args()

    gold_map = parse_qrels(Path(args.qrels))
    n_queries = len(gold_map)
    print(f"[score] qrels: {n_queries} queries with relevance judgments")

    runs_dir = Path(args.runs_dir)
    run_files = sorted(runs_dir.glob("*_run.txt"))
    if not run_files:
        raise SystemExit(f"No *_run.txt files in {runs_dir}")

    rows = []
    for rf in run_files:
        method = rf.stem.replace("_run", "")  # bm25, rerank
        run_map = parse_trec_run(rf)
        summary, _perq = evaluate_run(gold_map, run_map, ks_recall=RECALL_KS)
        br = BatchResult(method=method, batch=args.batch,
                         n_queries=n_queries, metrics=summary)
        rows.append(br.to_row())
        print(f"[score] {method}: MRR@10={summary['MRR@10']:.4f} "
              f"MAP@10={summary['MAP@10']:.4f} "
              f"MeanR@100={summary['MeanR@100']:.4f}")

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[score] wrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

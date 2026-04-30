#!/usr/bin/env python3
# Run the public ragnarok-style baseline end-to-end for the dicty 7a goldset:
#
#   topics.jsonl   --[Pyserini BM25 search]-->   bm25_run.txt
#                                                 |
#                                                 +-> rank_llm.ZephyrReranker
#                                                 -->  rerank_run.txt
#
# Outputs both TREC run files so we can score MRR@10/recall@K against the
# project's evaluator with a separate scoring script.
#
# Container: Dockerfile.ragnarok_baseline (rank_llm 0.25.7 + pyserini 0.44).

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from pyserini.search.lucene import LuceneSearcher

# rank_llm imports happen lazily to keep BM25-only mode runnable without GPU.


def load_topics(path: Path):
    topics = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            topics.append((str(d["qid"]), d["text"]))
    return topics


def write_trec_run(path: Path, run_name: str, ranked: dict):
    """ranked: {qid: [(docid, score), ...]} — write TREC 6-col format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, hits in ranked.items():
            for rank, (docid, score) in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")


def bm25_search(index_dir: Path, topics, top_k: int, threads: int):
    print(f"[bm25] LuceneSearcher({index_dir})", flush=True)
    searcher = LuceneSearcher(str(index_dir))
    print(f"[bm25] num_docs={searcher.num_docs}", flush=True)

    qids = [qid for qid, _ in topics]
    queries = [text for _, text in topics]

    print(f"[bm25] batch_search n_queries={len(queries)} k={top_k} threads={threads}", flush=True)
    t0 = time.time()
    batch = searcher.batch_search(
        queries=queries,
        qids=qids,
        k=top_k,
        threads=threads,
    )
    print(f"[bm25] batch_search done in {time.time()-t0:.1f}s", flush=True)

    ranked = {}
    for qid, _ in topics:
        hits = batch.get(qid, [])
        ranked[qid] = [(h.docid, float(h.score)) for h in hits]
    return ranked, searcher


def build_rank_llm_requests(topics, ranked_bm25, searcher):
    """Materialize rank_llm.data.Request objects (with raw doc text) for rerank."""
    from rank_llm.data import Candidate, Query, Request

    requests: List = []
    for qid, text in topics:
        cands = []
        for docid, score in ranked_bm25.get(qid, []):
            doc = searcher.doc(docid)
            raw = doc.raw() if doc is not None else None
            contents = ""
            if raw:
                try:
                    contents = json.loads(raw).get("contents", "") or ""
                except Exception:
                    contents = raw
            cands.append(
                Candidate(
                    docid=docid,
                    score=score,
                    doc={"contents": contents},
                )
            )
        requests.append(Request(query=Query(text=text, qid=qid), candidates=cands))
    return requests


def rerank_listwise(requests, model_path: str, context_size: int, num_passes: int,
                    window_size: int, step: int):
    from rank_llm.rerank.listwise import ZephyrReranker

    print(
        f"[rerank] ZephyrReranker model={model_path} ctx={context_size} "
        f"window={window_size} step={step} passes={num_passes}",
        flush=True,
    )
    reranker = ZephyrReranker(
        model_path=model_path,
        context_size=context_size,
        num_few_shot_examples=0,
    )

    t0 = time.time()
    reranked = reranker.rerank_batch(
        requests=requests,
        rank_end=len(requests[0].candidates) if requests else 100,
        window_size=window_size,
        step=step,
        num_passes=num_passes,
    )
    print(f"[rerank] done in {time.time()-t0:.1f}s", flush=True)
    return reranked


def reranked_to_dict(reranked) -> dict:
    out = {}
    for req in reranked:
        qid = req.query.qid
        hits = []
        # rank_llm sorts candidates by reranker score in-place; we re-emit ranks.
        for i, c in enumerate(req.candidates):
            # Use a descending pseudo-score so TREC tools order by the new rank.
            hits.append((c.docid, float(len(req.candidates) - i)))
        out[qid] = hits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True, help="topics.jsonl")
    ap.add_argument("--index", required=True, help="Pyserini Lucene index dir")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--bm25_top_k", type=int, default=100)
    ap.add_argument("--bm25_threads", type=int, default=8)

    ap.add_argument("--skip_rerank", action="store_true",
                    help="Run only BM25 (e.g. for quick smoke checks)")
    ap.add_argument("--rerank_model", default="castorini/rank_zephyr_7b_v1_full")
    ap.add_argument("--context_size", type=int, default=4096)
    ap.add_argument("--num_passes", type=int, default=3)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = load_topics(Path(args.topics))
    print(f"[main] loaded {len(topics)} topics", flush=True)

    ranked_bm25, searcher = bm25_search(
        Path(args.index), topics, args.bm25_top_k, args.bm25_threads
    )
    write_trec_run(out_dir / "bm25_run.txt", "bm25", ranked_bm25)
    print(f"[main] wrote {out_dir/'bm25_run.txt'}", flush=True)

    if args.skip_rerank:
        return

    requests = build_rank_llm_requests(topics, ranked_bm25, searcher)
    reranked = rerank_listwise(
        requests,
        model_path=args.rerank_model,
        context_size=args.context_size,
        num_passes=args.num_passes,
        window_size=args.window_size,
        step=args.step,
    )
    ranked_rr = reranked_to_dict(reranked)
    write_trec_run(out_dir / "rerank_run.txt", "rank_zephyr", ranked_rr)
    print(f"[main] wrote {out_dir/'rerank_run.txt'}", flush=True)


if __name__ == "__main__":
    main()

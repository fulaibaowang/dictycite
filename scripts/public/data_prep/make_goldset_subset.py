#!/usr/bin/env python3
"""
Create stratified train/test subsets from dicty_gold_llm_public.json.

Input/output use BioASQ-like schema (id, body, original_query, documents) so
scripts/public/shared_scripts/retrieval_eval/common.py can load the subsets. All other
metadata (group_claim_id, query, query_expand, docs, pmids, ...) is preserved.

Default behavior:
- Treat each item in "questions" as one query keyed by group_claim_id.
- Define minority as queries with >= 2 citations (len(docs) >= 2).
- Create disjoint train/test splits with a fixed seed.
- Slightly oversample minority while enforcing a minimum fraction.
- Write JSON outputs with the same schema plus a stats JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("Expected top-level dict with 'questions' key")
    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError("'questions' must be a list")
    return questions


def bucket_questions(
    questions: List[Dict[str, Any]],
    minority_min_citations: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    multiple_citations: List[Dict[str, Any]] = []
    one_citation: List[Dict[str, Any]] = []
    for q in questions:
        docs = q.get("docs", [])
        n_docs = len(docs) if isinstance(docs, list) else 0
        if n_docs >= minority_min_citations:
            multiple_citations.append(q)
        else:
            one_citation.append(q)
    return multiple_citations, one_citation


def compute_target_fraction(
    source_frac: float,
    min_frac: float,
    oversample_multiplier: float,
    max_frac: float,
) -> float:
    target = max(min_frac, source_frac * oversample_multiplier)
    return min(target, max_frac)


def allocate_minority_counts(
    total_minority: int,
    desired_train: int,
    desired_test: int,
) -> Tuple[int, int]:
    if total_minority <= 0:
        return 0, 0

    total_desired = desired_train + desired_test
    if total_desired <= total_minority:
        return desired_train, desired_test

    scale = total_minority / total_desired
    train_count = int(math.floor(desired_train * scale))
    test_count = int(math.floor(desired_test * scale))

    remaining = total_minority - (train_count + test_count)
    if remaining > 0:
        train_short = desired_train - train_count
        test_short = desired_test - test_count
        if train_short >= test_short:
            train_count += 1
            remaining -= 1
        if remaining > 0:
            test_count += 1
            remaining -= 1

    return train_count, test_count


def sample_split(
    minority: List[Dict[str, Any]],
    majority: List[Dict[str, Any]],
    train_size: int,
    test_size: int,
    seed: int,
    minority_target_fraction: float,
    min_minority_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)

    minority_shuffled = minority[:]
    majority_shuffled = majority[:]
    rng.shuffle(minority_shuffled)
    rng.shuffle(majority_shuffled)

    desired_train = int(math.ceil(train_size * max(min_minority_fraction, minority_target_fraction)))
    desired_test = int(math.ceil(test_size * max(min_minority_fraction, minority_target_fraction)))

    train_minority, test_minority = allocate_minority_counts(
        len(minority_shuffled),
        desired_train,
        desired_test,
    )

    test_minority_items = minority_shuffled[:test_minority]
    train_minority_items = minority_shuffled[test_minority : test_minority + train_minority]

    remaining_majority = majority_shuffled

    test_majority_needed = max(0, test_size - len(test_minority_items))
    test_majority_items = remaining_majority[:test_majority_needed]

    remaining_majority = remaining_majority[test_majority_needed:]

    train_majority_needed = max(0, train_size - len(train_minority_items))
    train_majority_items = remaining_majority[:train_majority_needed]

    test_items = test_minority_items + test_majority_items
    train_items = train_minority_items + train_majority_items

    rng.shuffle(test_items)
    rng.shuffle(train_items)

    if len(test_items) != test_size or len(train_items) != train_size:
        raise ValueError(
            f"Split size mismatch (train={len(train_items)}, test={len(test_items)})"
        )

    return train_items, test_items


def summarize(questions: List[Dict[str, Any]], minority_min_citations: int) -> Dict[str, Any]:
    total = len(questions)
    multiple_citations = 0
    one_citation = 0
    for q in questions:
        docs = q.get("docs", [])
        n_docs = len(docs) if isinstance(docs, list) else 0
        if n_docs >= minority_min_citations:
            multiple_citations += 1
        else:
            one_citation += 1
    frac = (multiple_citations / total) if total else 0.0
    return {
        "total": total,
        "multiple_citations": multiple_citations,
        "one_citation": one_citation,
        "multiple_citations_fraction": round(frac, 4),
    }


def write_json(path: Path, questions: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"questions": questions}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified goldset subsets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/cleaned/dicty_gold_llm_public.json"),
        help="Path to dicty_gold_llm_public.json",
    )
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-minority-fraction", type=float, default=0.10)
    parser.add_argument("--oversample-multiplier", type=float, default=1.25)
    parser.add_argument("--max-minority-fraction", type=float, default=0.50)
    parser.add_argument("--minority-min-citations", type=int, default=2)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("example/dicty_gold_llm_public_train_200.json"),
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("example/dicty_gold_llm_public_test_50.json"),
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("example/dicty_gold_llm_public_subset_stats.json"),
    )

    args = parser.parse_args()

    questions = load_questions(args.input)
    multiple_citations, one_citation = bucket_questions(questions, args.minority_min_citations)

    source_summary = summarize(questions, args.minority_min_citations)
    source_frac = source_summary["multiple_citations_fraction"]

    target_frac = compute_target_fraction(
        source_frac,
        args.min_minority_fraction,
        args.oversample_multiplier,
        args.max_minority_fraction,
    )

    train_items, test_items = sample_split(
        minority=multiple_citations,
        majority=one_citation,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        minority_target_fraction=target_frac,
        min_minority_fraction=args.min_minority_fraction,
    )

    write_json(args.train_out, train_items)
    write_json(args.test_out, test_items)

    stats = {
        "input": str(args.input),
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "multiple_citations_min": args.minority_min_citations,
        "min_multiple_citations_fraction": args.min_minority_fraction,
        "oversample_multiplier": args.oversample_multiplier,
        "max_multiple_citations_fraction": args.max_minority_fraction,
        "source": source_summary,
        "target_multiple_citations_fraction": round(target_frac, 4),
        "train": summarize(train_items, args.minority_min_citations),
        "test": summarize(test_items, args.minority_min_citations),
        "outputs": {
            "train": str(args.train_out),
            "test": str(args.test_out),
        },
    }

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Source:", source_summary)
    print("Target multiple citations fraction:", round(target_frac, 4))
    print("Train:", stats["train"])
    print("Test:", stats["test"])
    print("Saved:", args.train_out)
    print("Saved:", args.test_out)
    print("Saved stats:", args.stats_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enrich JSONL queries using a tabular entity file and YAML config.

Requires: pip install pyyaml  (see scripts/public/data_prep/requirements-query-expansion.txt)

Example (example train split has no expansion keys; see ``make_goldset_subset.py``):
  python scripts/public/data_prep/apply_query_expansion.py \\
    --config scripts/public/data_prep/conf/query_expansion_dicty_gene.yaml \\
    --table dictybase_files/gene_information.txt \\
    --input example/dicty_gold_llm_public_train_200.jsonl \\
    --output /tmp/train_200_expanded.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from query_expansion.config import load_expansion_config
from query_expansion.expand import expand_query_structured
from query_expansion.table import build_entity_index


def main() -> None:
    p = argparse.ArgumentParser(description="Apply YAML-driven query expansion to JSONL.")
    p.add_argument("--config", type=Path, required=True, help="Path to expansion YAML")
    p.add_argument("--table", type=Path, required=True, help="Entity TSV path")
    p.add_argument("--input", type=Path, required=True, help="Input JSONL")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL")
    p.add_argument(
        "--query-field",
        type=str,
        default=None,
        help="Override config query.read_field",
    )
    args = p.parse_args()

    cfg = load_expansion_config(args.config)
    if args.query_field:
        cfg.query.read_field = args.query_field

    index = build_entity_index(args.table, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    qf = cfg.query.read_field

    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSONL lines must be objects")
            text = obj.get(qf)
            if text is None:
                raise KeyError(f"missing query field {qf!r} in record keys={list(obj)!r}")

            for vk, spec in cfg.expand_variants.items():
                expanded, _, _ = expand_query_structured(str(text), index, vk)
                obj[spec.output_field] = expanded

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

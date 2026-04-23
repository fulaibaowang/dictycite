"""Smoke tests for query_expansion.

Expanded regression targets live next to the tracked subsets under ``example/`` as
``dicty_gold_llm_public_*_expanded.jsonl`` (gitignored). Generate once with::

  python3 scripts/public/data_prep/apply_query_expansion.py \\
    --config scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml \\
    --table dictybase_files/gene_information.txt \\
    --input example/dicty_gold_llm_public_train_200.jsonl \\
    --output example/dicty_gold_llm_public_train_200_expanded.jsonl

  python3 scripts/public/data_prep/apply_query_expansion.py \\
    --config scripts/public/data_prep/conf/query_expansion_dicty_gene.example.yaml \\
    --table dictybase_files/gene_information.txt \\
    --input example/dicty_gold_llm_public_test_50.jsonl \\
    --output example/dicty_gold_llm_public_test_50_expanded.jsonl

Run (no pytest required):
  python3 scripts/public/data_prep/query_expansion/test_smoke.py

Or with pytest if installed:
  pytest scripts/public/data_prep/query_expansion/test_smoke.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PREP = HERE.parent
if str(DATA_PREP) not in sys.path:
    sys.path.insert(0, str(DATA_PREP))
REPO = HERE.parents[3]

EXAMPLE = REPO / "example"
REF_TRAIN = EXAMPLE / "dicty_gold_llm_public_train_200_expanded.jsonl"
REF_TEST = EXAMPLE / "dicty_gold_llm_public_test_50_expanded.jsonl"

_SKIP_MSG = (
    "Expanded reference JSONL missing under example/ (gitignored). "
    "See module docstring for apply_query_expansion commands."
)


def _skip_if_no_refs() -> None:
    if REF_TRAIN.is_file() and REF_TEST.is_file():
        return
    if importlib.util.find_spec("pytest") is not None:
        import pytest

        pytest.skip(_SKIP_MSG)
    raise AssertionError(_SKIP_MSG)


def _paths_train() -> tuple[Path, Path, Path, Path]:
    cfg = DATA_PREP / "conf" / "query_expansion_dicty_gene.example.yaml"
    table = REPO / "dictybase_files" / "gene_information.txt"
    base = EXAMPLE / "dicty_gold_llm_public_train_200.jsonl"
    assert cfg.is_file() and table.is_file() and base.is_file()
    return cfg, table, base, REF_TRAIN


def test_expand_train_200_matches_reference() -> None:
    _skip_if_no_refs()
    from query_expansion.config import load_expansion_config
    from query_expansion.expand import expand_query_structured
    from query_expansion.table import build_entity_index

    dicty_cfg_path, gene_table, base_path, ref_path = _paths_train()
    cfg = load_expansion_config(dicty_cfg_path)
    index = build_entity_index(gene_table, cfg)

    with ref_path.open(encoding="utf-8") as fr, base_path.open(encoding="utf-8") as fb:
        for lr, lb in zip(fr, fb):
            ref = json.loads(lr)
            base = json.loads(lb)
            assert ref["query_id"] == base["query_id"]
            text = base[cfg.query.read_field]
            for vk in index.variant_keys_in_order():
                out_key = cfg.expansion_outputs[vk]
                got, _, _ = expand_query_structured(str(text), index, vk)
                assert got == ref[out_key], (ref["query_id"], out_key)


def test_expand_test_50_matches_reference() -> None:
    _skip_if_no_refs()
    from query_expansion.config import load_expansion_config
    from query_expansion.expand import expand_query_structured
    from query_expansion.table import build_entity_index

    dicty_cfg_path, gene_table, _, _ = _paths_train()
    cfg = load_expansion_config(dicty_cfg_path)
    index = build_entity_index(gene_table, cfg)
    base_path = EXAMPLE / "dicty_gold_llm_public_test_50.jsonl"
    assert base_path.is_file()

    with REF_TEST.open(encoding="utf-8") as fr, base_path.open(encoding="utf-8") as fb:
        for lr, lb in zip(fr, fb):
            ref = json.loads(lr)
            base = json.loads(lb)
            assert ref["query_id"] == base["query_id"]
            text = base[cfg.query.read_field]
            for vk in index.variant_keys_in_order():
                out_key = cfg.expansion_outputs[vk]
                got, _, _ = expand_query_structured(str(text), index, vk)
                assert got == ref[out_key], (ref["query_id"], out_key)


if __name__ == "__main__":
    if not (REF_TRAIN.is_file() and REF_TEST.is_file()):
        print(_SKIP_MSG, file=sys.stderr)
        sys.exit(0)
    test_expand_train_200_matches_reference()
    test_expand_test_50_matches_reference()
    print("ok")

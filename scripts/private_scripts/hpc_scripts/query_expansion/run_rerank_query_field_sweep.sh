#!/usr/bin/env bash
#
# Run retrieval once with BM25 and Dense fixed to query_text_expansion_long, then run the
# reranker 3 times with query-field query_text, query_text_expansion_synonyms, query_text_expansion_long.
# Requires DOCS_JSONL in config (reranker needs it).
#
# Usage:
#   From repo root: ./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config scripts/private_scripts/hpc_scripts/config.env
#
# Output layout: under BASE_OUT (fixed_long_rerank_sweep/ when WORKFLOW_SWEEP_OUTPUT_DIR is set):
#   .../fixed_long_rerank_sweep/retrieval/{bm25,dense,fusion}/, rerank_body/, rerank_synonyms/, rerank_long/
# When WORKFLOW_SWEEP_OUTPUT_DIR is set (e.g. by sbatch), BASE_OUT = .../fixed_long_rerank_sweep (no overlap with query_field_sweep).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PIPELINE="$REPO_ROOT/scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh"
PIPELINE_SCRIPT_DIR="$(cd "$(dirname "$PIPELINE")" && pwd)"

CONFIG_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--config)
      [ -z "${2:-}" ] && { echo "Error: --config requires a path." >&2; exit 1; }
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--config|-c <config.env>]"
      echo "  Runs retrieval once (BM25+Dense+Hybrid with query_text_expansion_long), then reranker 3 times (query_text, synonyms, long)."
      echo "  -c, --config PATH   Source PATH before running (required unless already sourced)."
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (use -h for help)" >&2
      exit 1
      ;;
  esac
done

if [ -n "$CONFIG_FILE" ]; then
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  echo "Using config: $CONFIG_FILE"
fi

QF_BODY="query_text"
QF_SYNONYMS="query_text_expansion_synonyms"
QF_LONG="query_text_expansion_long"

# ----- 1) Run retrieval once (BM25 + Dense + Hybrid, both use long) -----
if [ -n "$CONFIG_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi

# Use WORKFLOW_SWEEP_OUTPUT_DIR if set (sbatch sets it); rerank sweep always uses fixed_long_rerank_sweep/ subfolder.
if [ -n "${WORKFLOW_SWEEP_OUTPUT_DIR:-}" ]; then
  BASE_OUT="${WORKFLOW_SWEEP_OUTPUT_DIR}/fixed_long_rerank_sweep"
else
  BASE_OUT="${WORKFLOW_OUTPUT_DIR:-$REPO_ROOT/output/workflow_hpc_test}"
fi
export WORKFLOW_OUTPUT_DIR="$BASE_OUT"

echo ""
echo "========== Retrieval (BM25=long, Dense=long) -> $WORKFLOW_OUTPUT_DIR =========="
mkdir -p "$WORKFLOW_OUTPUT_DIR"
"$PIPELINE" --no-rerank --bm25-query-field "$QF_LONG" --dense-query-field "$QF_LONG"

if [ -z "${DOCS_JSONL:-}" ]; then
  echo "DOCS_JSONL not set; skipping reranker sweep. Set DOCS_JSONL in config to run reranker."
  echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (retrieval/{bm25,dense,fusion}/)"
  exit 0
fi

# ----- 2) Reranker sweep over 3 query fields -----
# Re-source config so we have RERANK_*, TOP_K, etc.
if [ -n "$CONFIG_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
  export WORKFLOW_OUTPUT_DIR="$BASE_OUT"
fi

# Must match run_retrieval_rerank_pipeline.sh: retrieval/fusion (not a top-level "hybrid/" dir).
HYBRID_OUT="$WORKFLOW_OUTPUT_DIR/retrieval/fusion"
TOP_K="${TOP_K:-5000}"
RECALL_KS="${RECALL_KS:-50,100,200,300,400,500,1000,2000,5000}"
HYBRID_CAP="${HYBRID_CAP:-$TOP_K}"
RERANK_CANDIDATE_LIMIT="${RERANK_CANDIDATE_LIMIT:-$TOP_K}"

if [ "$RERANK_CANDIDATE_LIMIT" -le "$HYBRID_CAP" ]; then
  RERANK_RAW=$RERANK_CANDIDATE_LIMIT
else
  RERANK_RAW=$HYBRID_CAP
fi
if [ "$RERANK_RAW" -lt 30 ]; then
  RERANK_EFFECTIVE=30
elif [ "$RERANK_RAW" -gt 2000 ]; then
  RERANK_EFFECTIVE=2000
else
  RERANK_EFFECTIVE=$RERANK_RAW
fi

if [ -f "$DOCS_JSONL" ]; then
  DOCS_JSONL_LINES=$(wc -l < "$DOCS_JSONL" 2>/dev/null || echo 0)
  if [ "$DOCS_JSONL_LINES" -gt 0 ] && [ "$DOCS_JSONL_LINES" -lt 2000 ] && [ "$DOCS_JSONL_LINES" -lt "$RERANK_EFFECTIVE" ]; then
    RERANK_EFFECTIVE=$DOCS_JSONL_LINES
    echo "Reranker candidate-limit capped to doc count in DOCS_JSONL ($DOCS_JSONL_LINES)"
  fi
fi

export PYTHONPATH="$PIPELINE_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

declare -a RERANK_QF_FULL=("$QF_BODY" "$QF_SYNONYMS" "$QF_LONG")
declare -a RERANK_QF_SHORT=("body" "synonyms" "long")

for i in 0 1 2; do
  QF_FULL="${RERANK_QF_FULL[$i]}"
  QF_SHORT="${RERANK_QF_SHORT[$i]}"
  RERANK_OUT="$WORKFLOW_OUTPUT_DIR/rerank_$QF_SHORT"
  echo ""
  if [ -f "$RERANK_OUT/metrics.csv" ] || [ -n "$(find "$RERANK_OUT" -maxdepth 2 -name '*.tsv' 2>/dev/null | head -1)" ]; then
    echo "========== Reranker (query-field=$QF_FULL) -> $RERANK_OUT (skip: output exists) =========="
    continue
  fi
  echo "========== Reranker (query-field=$QF_FULL) -> $RERANK_OUT =========="
  mkdir -p "$RERANK_OUT"
  RERANK_ARGS=(
    --runs-dir "$HYBRID_OUT/runs"
    --output-dir "$RERANK_OUT"
    --docs-jsonl "$DOCS_JSONL"
    --candidate-limit "$RERANK_EFFECTIVE"
    --ks-recall "${RERANK_KS_RECALL:-$RECALL_KS}"
    --query-field "$QF_FULL"
  )
  [ -n "${TRAIN_JSON:-}" ] && RERANK_ARGS+=(--train-json "$TRAIN_JSON")
  [ -n "${TEST_BATCH_JSONS:-}" ] && RERANK_ARGS+=(--test_batch_jsons $TEST_BATCH_JSONS)
  [ -n "${RERANK_MODEL:-}" ] && RERANK_ARGS+=(--model "$RERANK_MODEL")
  [ -n "${RERANK_MODEL_DEVICE:-}" ] && RERANK_ARGS+=(--model-device "$RERANK_MODEL_DEVICE")
  [ -n "${RERANK_MODEL_BATCH:-}" ] && RERANK_ARGS+=(--model-batch "$RERANK_MODEL_BATCH")
  [ -n "${RERANK_MODEL_MAX_LENGTH:-}" ] && RERANK_ARGS+=(--model-max-length "$RERANK_MODEL_MAX_LENGTH")
  [ "${RERANK_DISABLE_METRICS:-0}" = "1" ] && RERANK_ARGS+=(--disable-metrics)
  [ "${RERANK_USE_MULTI_GPU:-0}" = "1" ] && RERANK_ARGS+=(--use-multi-gpu)
  [ -n "${RERANK_NUM_GPUS:-}" ] && RERANK_ARGS+=(--num-gpus "$RERANK_NUM_GPUS")
  python "$PIPELINE_SCRIPT_DIR/rerank/rerank_crossencoder.py" "${RERANK_ARGS[@]}"
done

echo ""
echo "Done. Outputs: $WORKFLOW_OUTPUT_DIR (retrieval/{bm25,dense,fusion}/, rerank_{body,synonyms,long}/)"

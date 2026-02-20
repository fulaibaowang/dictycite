#!/usr/bin/env bash
#
# Run BM25 + Dense + Hybrid for all 3×3 query-field combinations (body, body_expansion_synonyms, body_expansion_long).
# Stops at hybrid; no reranker. Each combination gets its own WORKFLOW_OUTPUT_DIR subdir.
#
# Usage:
#   ./run_query_field_sweep.sh --config scripts/private_scripts/hpc_scripts/config.env
#   source config.env && ./run_query_field_sweep.sh
#
# Output layout (if WORKFLOW_OUTPUT_DIR is output/workflow_hpc_test):
#   output/workflow_hpc_test/bm25_body_dense_body/    (bm25/, dense/, hybrid/)
#   output/workflow_hpc_test/bm25_body_dense_synonyms/
#   ... (9 dirs total)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PIPELINE="$REPO_ROOT/scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh"

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
      echo "  Runs 9 pipeline runs (3 BM25 query-fields × 3 Dense query-fields) up to hybrid (no rerank)."
      echo "  -c, --config PATH   Source PATH before each run (required unless already sourced)."
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

# Query-field options (full values for --query-field)
QF_BODY="body"
QF_SYNONYMS="body_expansion_synonyms"
QF_LONG="body_expansion_long"

# Short names for directory naming
declare -A QF_SHORT
QF_SHORT[$QF_BODY]=body
QF_SHORT[$QF_SYNONYMS]=synonyms
QF_SHORT[$QF_LONG]=long

BM25_OPTS=("$QF_BODY" "$QF_SYNONYMS" "$QF_LONG")
DENSE_OPTS=("$QF_BODY" "$QF_SYNONYMS" "$QF_LONG")

COMBO=0
TOTAL=9
for BM25_Q in "${BM25_OPTS[@]}"; do
  for DENSE_Q in "${DENSE_OPTS[@]}"; do
    COMBO=$((COMBO + 1))
    BM25_S="${QF_SHORT[$BM25_Q]}"
    DENSE_S="${QF_SHORT[$DENSE_Q]}"
    SUBDIR="bm25_${BM25_S}_dense_${DENSE_S}"

    if [ -n "$CONFIG_FILE" ]; then
      set -a
      # shellcheck source=/dev/null
      source "$CONFIG_FILE"
      set +a
    fi

    # Override: one output dir per combination (sibling to default or under it)
    BASE_OUT="${WORKFLOW_OUTPUT_DIR:-$REPO_ROOT/output/workflow_hpc_test}"
    export WORKFLOW_OUTPUT_DIR="$BASE_OUT/$SUBDIR"

    echo ""
    echo "========== [$COMBO/$TOTAL] BM25=$BM25_Q Dense=$DENSE_Q -> $WORKFLOW_OUTPUT_DIR =========="
    mkdir -p "$WORKFLOW_OUTPUT_DIR"
    # Pass query fields as args so the pipeline always uses the right ones (env can be lost e.g. under job runners)
    "$PIPELINE" --no-rerank --bm25-query-field "$BM25_Q" --dense-query-field "$DENSE_Q"
  done
done

echo ""
echo "Done. 9 runs written under $BASE_OUT (bm25_*_dense_*)."

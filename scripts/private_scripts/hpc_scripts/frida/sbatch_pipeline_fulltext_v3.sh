#!/bin/bash
# Retrieval + rerank pipeline for the full-text v3 comparison (abstract-only vs abstract + v3
# full-text chunks, with and without gene-aware query expansion). One script for every arm:
# corpus, indexes, query field and reranker come from the config.
#
# Submit through the provenance wrapper (never plain sbatch):
#   ~/hpc-tools/scripts/hpc_submit.sh scripts/private_scripts/hpc_scripts/frida/sbatch_pipeline_fulltext_v3.sh \
#       scripts/private_scripts/hpc_scripts/frida/config_frida_fulltext_v3_<arm>.env
#
# The v3 indexes and corpus are not in the repo: link them into the checkout's gitignored
# dicty_simulated_data/indexes/ and dicty_fulltext_corpus/v3/ before submitting.
#
#SBATCH -J dicty_fulltext_v3
#SBATCH -p frida
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PIPELINE_CONFIG="${HPC_CONFIG:-${PIPELINE_CONFIG:?set PIPELINE_CONFIG or submit through hpc_submit}}"
[[ -f "$PIPELINE_CONFIG" ]] || { echo "ERROR: config not found: $PIPELINE_CONFIG" >&2; exit 1; }
echo "[config] sourcing $PIPELINE_CONFIG (RUN_ID=${HPC_RUN_ID:-unset})"
source "$PIPELINE_CONFIG"

REPO="${REPO:-$HOME/dictycite}"
CONTAINER_IMG="${CONTAINER_IMG:?set CONTAINER_IMG (.sqfs image)}"
WORKSPACE="${WORKSPACE:?set WORKSPACE (host dir mounted as /data)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR (container path under /data)}"
PUBMED_HOST="${PUBMED_HOST:-/shared/workspace/biolab/pubmed}"

cd "$REPO"; mkdir -p logs; WORKDIR="$PWD"
# The config is sourced again inside the container (it sets WORKFLOW_OUTPUT_DIR etc.), so it
# must be reachable there: a path relative to the checkout, or absolute inside it.
case "$PIPELINE_CONFIG" in
  "$WORKDIR"/*) CTR_CONFIG="/work/${PIPELINE_CONFIG#"$WORKDIR"/}";;
  /*) echo "ERROR: config must be inside $WORKDIR: $PIPELINE_CONFIG" >&2; exit 1;;
  *)  CTR_CONFIG="$PIPELINE_CONFIG";;
esac
echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi -L 2>&1 || true

srun \
  --container-image="$CONTAINER_IMG" \
  --container-mount-home \
  --container-mounts "$WORKDIR:/work,$WORKSPACE:/data,$PUBMED_HOST:/pubmed" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail
    export HF_HOME=/pubmed/_hf_cache HF_HUB_CACHE=/pubmed/_hf_cache/hub
    export TRANSFORMERS_CACHE=/pubmed/_hf_cache/transformers
    export SENTENCE_TRANSFORMERS_HOME=/pubmed/_hf_cache/sentence_transformers
    export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TQDM_DISABLE=1

    python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'CUDA unavailable in this job step'
x = torch.randn(1, device='cuda'); torch.cuda.synchronize()
print('[probe] CUDA OK', torch.cuda.get_device_name(0))
PY

    source '${CTR_CONFIG}'
    for f in \"\$INPUT_JSONL\" \"\$DOCS_JSONL\" \"\$BM25_INDEX_PATH\" \"\$DENSE_INDEX_DIR\"; do
      [ -e \"\$f\" ] || { echo \"[error] not found: \$f\" >&2; exit 2; }
    done
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"
    cp '${CTR_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    if [ -n \"\${RETRIEVAL_COPY_FROM:-}\" ]; then
      [ -d \"\$RETRIEVAL_COPY_FROM/retrieval\" ] || { echo \"[error] no retrieval/ in \$RETRIEVAL_COPY_FROM\" >&2; exit 2; }
      echo \"[seed] retrieval from \$RETRIEVAL_COPY_FROM\"
      rm -rf \"\$WORKFLOW_OUTPUT_DIR/retrieval\"
      cp -a \"\$RETRIEVAL_COPY_FROM/retrieval\" \"\$WORKFLOW_OUTPUT_DIR/\"
    fi

    echo \"[indexes] BM25=\$BM25_INDEX_PATH DENSE=\$DENSE_INDEX_DIR DOCS=\$DOCS_JSONL\"
    echo \"[query] BM25=\${BM25_QUERY_FIELD:-query_text} DENSE=\${DENSE_QUERY_FIELD:-query_text} RERANK=\${RERANK_QUERY_FIELD:-query_text}\"
    sha256sum \"\$DOCS_JSONL\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${CTR_CONFIG}'
    echo \"[done] outputs in \$WORKFLOW_OUTPUT_DIR\"
  "
echo "Finished job ${SLURM_JOB_ID:-local} at $(date)"

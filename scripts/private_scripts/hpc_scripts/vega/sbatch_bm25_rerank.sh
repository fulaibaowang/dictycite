#!/bin/bash
#SBATCH -J dicty_ragnarok_bm25_rerank
#SBATCH -p gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# Public ragnarok-style baseline on the dicty 7a goldset:
#   topics.jsonl --[Pyserini BM25 top-100]--> ZephyrReranker (listwise)
#
# Prereq: sbatch_build_anserini_index.sh has been run (indexes/dicty_anserini_idx).
# Container: dictycite-ragnarok.sif (Dockerfile.ragnarok_baseline).
# Walltime estimate: ~3-6h for 1,656 queries x top-100 x num_passes=3 on one GPU.

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/ragnarok_2026.04.29.sif"
WORKDIR="${PWD}"

YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

GOLDSET_JSONL="output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl"
OUT_DIR="output/ragnarok_baseline"
TOPICS_JSONL="${OUT_DIR}/topics.jsonl"
QRELS_TXT="${OUT_DIR}/qrels.txt"
ANSERINI_INDEX_DIR="indexes/dicty_anserini_idx"
RUNS_DIR="${OUT_DIR}/runs"

BM25_TOP_K="${BM25_TOP_K:-100}"
NUM_PASSES="${NUM_PASSES:-3}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"
STEP="${STEP:-10}"
RERANK_MODEL="${RERANK_MODEL:-castorini/rank_zephyr_7b_v1_full}"

echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Container: ${CONTAINER_IMG}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

module purge
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

APPTAINER_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  APPTAINER_GPU_ARGS+=(--nv)
else
  echo "No GPUs allocated; rerank stage will fail. Run with --gres=gpu:1." >&2
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

singularity exec \
  "${APPTAINER_GPU_ARGS[@]}" \
  -B "${WORKDIR}:/work" \
  -B "${YUN_HOST}:/yun" \
  -B "${HOME_HOST}:/home/wangy" \
  --pwd /work \
  "${CONTAINER_IMG}" \
  bash -lc "
    set -euo pipefail

    export HF_HOME='/yun/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1
    export TQDM_DISABLE=1

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    mkdir -p '${OUT_DIR}' '${RUNS_DIR}'

    echo '[step 1/3] Export goldset -> topics.jsonl + qrels.txt'
    python3 scripts/public/ragnarok_baseline/export_goldset_topics.py \
      --input '${GOLDSET_JSONL}' \
      --topics_out '${TOPICS_JSONL}' \
      --qrels_out '${QRELS_TXT}'

    echo '[step 2/3] BM25 search + listwise rerank'
    python3 scripts/public/ragnarok_baseline/run_bm25_rerank.py \
      --topics '${TOPICS_JSONL}' \
      --index '${ANSERINI_INDEX_DIR}' \
      --output_dir '${RUNS_DIR}' \
      --bm25_top_k '${BM25_TOP_K}' \
      --bm25_threads 8 \
      --rerank_model '${RERANK_MODEL}' \
      --num_passes '${NUM_PASSES}' \
      --window_size '${WINDOW_SIZE}' \
      --step '${STEP}'

    echo '[step 3/3] Score MRR@10 + recall@K'
    python3 scripts/public/ragnarok_baseline/score_ragnarok_run.py \
      --qrels '${QRELS_TXT}' \
      --runs_dir '${RUNS_DIR}' \
      --output_csv '${OUT_DIR}/metrics.csv'

    echo '[done] ragnarok baseline complete'
    echo '--- metrics.csv ---'
    cat '${OUT_DIR}/metrics.csv'
  "

echo "Finished job ${SLURM_JOB_ID:-local} at $(date)"

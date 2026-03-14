#!/bin/bash
#SBATCH -J dense_medembed
#SBATCH -p dev
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/dictycite
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
WORKDIR="${PWD}"

JSONL_GLOB="/work/output/cleaned/articles_all_cleaned_abstract.jsonl"
OUT_DIR="/work/indexes/dicty_medembed_index"

# -----------------------------
# Params
# -----------------------------
BATCH_SIZE=256
M=32
EF_CONSTRUCTION=200
EF_SEARCH=100

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "jsonl_glob=${JSONL_GLOB}  out_dir=${OUT_DIR}"

# -----------------------------
# Run inside container
# -----------------------------
srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail
    python -u scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
      --jsonl_glob '${JSONL_GLOB}' \
      --out_dir '${OUT_DIR}' \
      --device 'cuda' \
      --batch_size ${BATCH_SIZE} \
      --M ${M} \
      --ef_construction ${EF_CONSTRUCTION} \
      --ef_search ${EF_SEARCH} \
      --dedup_pmids
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

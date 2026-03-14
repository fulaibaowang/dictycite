#!/bin/bash
#SBATCH -J bm25_index
#SBATCH -p amd
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd ~/dictycite
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs"
WORKDIR="${PWD}"
#PUBMED_HOST="/shared/workspace/biolab/pubmed"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "index_path=/work/indexes/dicty_bm25_index"

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
    python -u scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
      --jsonl_glob '/work/output/cleaned/articles_all_cleaned_abstract.jsonl' \
      --index_path '/work/indexes/dicty_bm25_index' \
      --threads 16 \
      --overwrite
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

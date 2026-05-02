#!/bin/bash
#SBATCH -J dicty_vega_bm25_idx
#SBATCH -p cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# PyTerrier BM25 index from the gold-build article corpus (7c), not
# output/cleaned/articles_all_cleaned_abstract.jsonl. The 7c JSONL exposes
# `abstract` so build_bm25_index_from_jsonl_shards.py indexes title+abstract;
# the older cleaned corpus often only had abstract_clean, which d.get("abstract")
# missed (title-only postings).

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

JSONL_GLOB="/work/output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl"
INDEX_PATH="/work/indexes/dicty_bm25_index"

echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Container: ${CONTAINER_IMG}"
echo "jsonl_glob=${JSONL_GLOB}  index_path=${INDEX_PATH}"

module purge
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

srun --mpi=none singularity exec \
  --cleanenv \
  -B "${WORKDIR}:/work" \
  -B "${YUN_HOST}:/yun" \
  -B "${HOME_HOST}:/home/wangy" \
  --pwd /work \
  "${CONTAINER_IMG}" \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1

    python -u scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
      --jsonl_glob '${JSONL_GLOB}' \
      --index_path '${INDEX_PATH}' \
      --threads 16 \
      --overwrite
  "

echo "Finished job ${SLURM_JOB_ID:-local} at $(date)"

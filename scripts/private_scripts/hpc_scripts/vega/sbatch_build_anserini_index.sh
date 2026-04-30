#!/bin/bash
#SBATCH -J dicty_ragnarok_anserini_idx
#SBATCH -p cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Build a Pyserini/Anserini Lucene index of the dicty corpus
# (output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl) for the
# ragnarok-style BM25 baseline.
# CPU-only — no GPU needed for indexing 20k docs.

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/ragnarok_2026.04.29.sif"
WORKDIR="${PWD}"

YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

INPUT_JSONL="output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl"
ANSERINI_INPUT_DIR="indexes/anserini_input/dicty"
ANSERINI_INDEX_DIR="indexes/dicty_anserini_idx"
SHARD_SIZE="${SHARD_SIZE:-50000}"

echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Container: ${CONTAINER_IMG}"

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

    echo '[step 1/2] Convert dicty JSONL -> Anserini JsonCollection shards'
    python3 scripts/public/ragnarok_baseline/convert_dicty_to_anserini.py \
      --input '${INPUT_JSONL}' \
      --output_dir '${ANSERINI_INPUT_DIR}' \
      --shard_size '${SHARD_SIZE}'

    echo '[step 2/2] Build Lucene index via Pyserini'
    rm -rf '${ANSERINI_INDEX_DIR}'
    mkdir -p '${ANSERINI_INDEX_DIR}'
    python3 -m pyserini.index.lucene \
      --collection JsonCollection \
      --input '${ANSERINI_INPUT_DIR}' \
      --index '${ANSERINI_INDEX_DIR}' \
      --generator DefaultLuceneDocumentGenerator \
      --threads 8 \
      --storePositions --storeDocvectors --storeRaw

    echo '[done] Index ready at ${ANSERINI_INDEX_DIR}'
    python3 -c \"
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('${ANSERINI_INDEX_DIR}')
print('num_docs:', s.num_docs)
\"
  "

echo "Finished job ${SLURM_JOB_ID:-local} at $(date)"

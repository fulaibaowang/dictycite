#!/bin/bash
#SBATCH -J rerank_sweep
#SBATCH -p dev
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/dictycite
mkdir -p logs

# -----------------------------
# Paths / inputs
# -----------------------------
CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26.sqfs"
WORKDIR="${PWD}"

PUBMED_HOST="/shared/workspace/biolab/pubmed"

# Config: same as full pipeline (must have DOCS_JSONL, indexes, etc.)
PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/fulltest/config_full.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running rerank query-field sweep with config: ${PIPELINE_CONFIG}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

if [[ "${NUM_GPUS}" -eq 0 ]]; then
  export ENROOT_DISABLE_NVIDIA=1
  export NVIDIA_VISIBLE_DEVICES=void
  echo "No GPUs allocated; disabling NVIDIA hooks for container"
fi

# -----------------------------
# Run inside container
# -----------------------------
srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail

    export HF_HOME='/pubmed/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export TRANSFORMERS_CACHE=\"\$HF_HOME/transformers\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1
    export TQDM_DISABLE=1

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    source '${PIPELINE_CONFIG}'
    export WORKFLOW_SWEEP_OUTPUT_DIR=\"\${REPO_ROOT}/output/workflow_baseline_full_sweep\"
    mkdir -p \"\$WORKFLOW_SWEEP_OUTPUT_DIR\"
    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_SWEEP_OUTPUT_DIR/\"

    echo \"[run] Rerank query-field sweep (retrieval once + rerank body/synonyms/long)\"
    ./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Rerank sweep completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

#!/bin/bash
#SBATCH -J dicty_pipeline_qe
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# VEGA: same pattern as BioASQ scripts/private_scripts/hpc/vega/sbatch_pipeline.sh
# (Apptainer .sif, bind mounts, HF cache on /yun).

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/vega/config_full_query_expansion.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running pipeline script with config: ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

module purge
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

APPTAINER_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  APPTAINER_GPU_ARGS+=(--nv)
else
  echo "No GPUs allocated; running container without --nv"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

srun --mpi=none singularity exec \
  --cleanenv \
  "${APPTAINER_GPU_ARGS[@]}" \
  -B "${WORKDIR}:/work" \
  -B "${PUBMED_HOST}:/pubmed" \
  -B "${YUN_HOST}:/yun" \
  -B "${HOME_HOST}:/home/wangy" \
  --pwd /work \
  "${CONTAINER_IMG}" \
  bash -lc "
    set -euo pipefail

    export HF_HOME='/yun/_hf_cache'
    export HF_HUB_CACHE=\"\$HF_HOME/hub\"
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$HF_HOME/transformers\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1
    export TQDM_DISABLE=1

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    source '${PIPELINE_CONFIG}'
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"
    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    echo \"[run] Starting retrieval + rerank + evidence + generation pipeline\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Pipeline completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

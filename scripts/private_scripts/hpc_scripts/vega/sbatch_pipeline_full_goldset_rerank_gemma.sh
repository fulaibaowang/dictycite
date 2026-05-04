#!/bin/bash
#SBATCH -J dicty_pipe_7a_gold_rerank_gemma
#SBATCH -p gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# Full public goldset with BAAI/bge-reranker-v2-gemma (LLM / FlagLLMReranker backend).
# Config: config_vega_7a_public_goldset_rerank_gemma.env
#
# To reuse BM25+dense+fusion from an existing run, set in the env file:
#   RETRIEVAL_COPY_FROM=/home/wangy/dictycite_output/workflow_vega_7a_public_goldset_both_routes
# (uncomment the line). The script rsyncs only retrieval/ into this job's WORKFLOW_OUTPUT_DIR.
#
# Index prerequisite: same as sbatch_pipeline_full_goldset.sh (BM25 + dense under REPO_ROOT/indexes/).

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/vega/config_vega_7a_public_goldset_rerank_gemma.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running pipeline script with config: ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"
echo "Optional retrieval seed: set RETRIEVAL_COPY_FROM in ${PIPELINE_CONFIG} to copy retrieval/ from a prior run"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "[host] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[host] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "[host] /dev/nvidia* listing:"
ls -l /dev/nvidia* 2>&1 || true
echo "[host] nvidia-smi -L:"
nvidia-smi -L 2>&1 || true

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

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

singularity exec \
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

    echo \"[ctr] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    echo \"[ctr] /dev/nvidia* listing inside container:\"
    ls -l /dev/nvidia* 2>&1 || true
    echo \"[ctr] nvidia-smi -L:\"
    nvidia-smi -L || true

    echo \"[probe] validating CUDA runtime before retrieval pipeline\"
    if ! python - <<'PY'
import torch
print(f'[probe] torch={torch.__version__}')
print(f'[probe] cuda_available={torch.cuda.is_available()}')
print(f'[probe] device_count={torch.cuda.device_count()}')
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('CUDA unavailable in this job step')
x = torch.randn(1, device='cuda'); y = x * 2; del x, y
torch.cuda.synchronize()
print('[probe] CUDA allocation/synchronize OK')
PY
    then
      echo \"[probe] CUDA failed at the start of the pipeline; aborting before retrieval/rerank.\" >&2
      echo \"[probe] Likely a stuck GPU on this node. Resubmit (Slurm may pick a different GPU)\" >&2
      echo \"[probe] or use sbatch --exclude=<node> to avoid that node.\" >&2
      exit 42
    fi

    source '${PIPELINE_CONFIG}'
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"

    if [ -n \"\${RETRIEVAL_COPY_FROM:-}\" ] && [ -d \"\${RETRIEVAL_COPY_FROM}/retrieval\" ]; then
      echo \"[seed] rsync retrieval: \${RETRIEVAL_COPY_FROM}/retrieval/ -> \${WORKFLOW_OUTPUT_DIR}/retrieval/\"
      mkdir -p \"\${WORKFLOW_OUTPUT_DIR}/retrieval\"
      rsync -a \"\${RETRIEVAL_COPY_FROM}/retrieval/\" \"\${WORKFLOW_OUTPUT_DIR}/retrieval/\"
    else
      echo \"[seed] RETRIEVAL_COPY_FROM unset or no retrieval/ there — running full retrieval (steps 1–3)\"
    fi

    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    echo \"[indexes] BM25_INDEX_PATH=\${BM25_INDEX_PATH}\"
    echo \"[indexes] DENSE_INDEX_DIR=\${DENSE_INDEX_DIR}\"
    echo \"[indexes] DOCS_JSONL=\${DOCS_JSONL}\"

    echo \"[run] Starting retrieval + rerank (Gemma) + evidence + generation pipeline\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Pipeline completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

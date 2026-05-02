#!/bin/bash
#SBATCH -J dicty_pipeline_7a_goldset
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# Full public goldset (7a_dicty_gold_llm_public.jsonl), not the example train_200/test_50 subset.
# Same container/bind pattern as sbatch_pipeline.sh; extend --time if the run exceeds the wall clock.
#
# Index prerequisite: config_vega_7a_public_goldset.env points BM25_INDEX_PATH and DENSE_INDEX_DIR
# at indexes/dicty_bm25_index and indexes/dicty_medembed_index. Those must be built from
# output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl (real `abstract` field) so first-stage
# retrieval matches DOCS_JSONL. Rebuild on Vega: sbatch_vega_bm25_index.sh, then sbatch_vega_dense_index.sh.

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/vega/config_vega_7a_public_goldset.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Running pipeline script with config: ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"
echo "Index rebuild (7c corpus): scripts/private_scripts/hpc_scripts/vega/sbatch_vega_bm25_index.sh"
echo "                            scripts/private_scripts/hpc_scripts/vega/sbatch_vega_dense_index.sh"

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

# ---------- Vega GPU binding (see sbatch_vega_dense_index.sh for rationale) ----------
# USE_NVCCLI=1 (default): libnvidia-container-cli with NVIDIA_VISIBLE_DEVICES.
# USE_NVCCLI=0: legacy --nv with auxiliary device binds + only allocated /dev/nvidiaN.
USE_NVCCLI="${USE_NVCCLI:-1}"
ALLOC_GPU_IDS="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}"

APPTAINER_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  if [[ "${USE_NVCCLI}" == "1" ]]; then
    APPTAINER_GPU_ARGS+=(--nvccli)
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${ALLOC_GPU_IDS:-all}"
    echo "[gpu] Using --nvccli  NVIDIA_VISIBLE_DEVICES=${APPTAINERENV_NVIDIA_VISIBLE_DEVICES}"
  else
    APPTAINER_GPU_ARGS+=(--nv)
    echo "[gpu] Using --nv (legacy)"
    for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
      [[ -e "$d" ]] && APPTAINER_GPU_ARGS+=(-B "$d")
    done
    if [[ -n "${ALLOC_GPU_IDS}" ]]; then
      IFS=',' read -ra _ids <<< "${ALLOC_GPU_IDS}"
      for id in "${_ids[@]}"; do
        if [[ "$id" =~ ^[0-9]+$ && -e "/dev/nvidia${id}" ]]; then
          APPTAINER_GPU_ARGS+=(-B "/dev/nvidia${id}")
        fi
      done
    fi
  fi
else
  echo "No GPUs allocated; running container without GPU args"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

# Forward Slurm-provided GPU mapping; avoid --cleanenv pitfall (drops CUDA_VISIBLE_DEVICES).
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
      echo \"[probe] If --nvccli was selected, ensure libnvidia-container-cli is available;\" >&2
      echo \"[probe] otherwise retry with USE_NVCCLI=0 or resubmit to a different GPU node.\" >&2
      exit 42
    fi

    source '${PIPELINE_CONFIG}'
    mkdir -p \"\$WORKFLOW_OUTPUT_DIR\"
    cp '${PIPELINE_CONFIG}' \"\$WORKFLOW_OUTPUT_DIR/\"

    echo \"[indexes] BM25_INDEX_PATH=\${BM25_INDEX_PATH}\"
    echo \"[indexes] DENSE_INDEX_DIR=\${DENSE_INDEX_DIR}\"
    echo \"[indexes] DOCS_JSONL=\${DOCS_JSONL}\"

    echo \"[run] Starting retrieval + rerank + evidence + generation pipeline (7a full goldset)\"
    ./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config '${PIPELINE_CONFIG}'

    echo \"[done] Pipeline completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

#!/bin/bash
#SBATCH -J dicty_vega_dense_medembed
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# HNSW dense index (MedEmbed / sentence-transformers) from the gold-build
# article corpus (7c). Same rationale as sbatch_vega_bm25_index.sh: 7c stores
# text under `abstract`, which build_dense_hnsw_index_from_jsonl_shards.py
# reads the `abstract` field (not abstract_clean).

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

JSONL_GLOB="/work/output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl"
OUT_DIR="/work/indexes/dicty_medembed_index"

BATCH_SIZE="${BATCH_SIZE:-256}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
EF_SEARCH="${EF_SEARCH:-100}"
DENSE_DEVICE="${DENSE_DEVICE:-cuda}"            # cuda|cpu
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-0}"   # 1 => fallback to CPU when CUDA probe fails

echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Container: ${CONTAINER_IMG}"
echo "jsonl_glob=${JSONL_GLOB}  out_dir=${OUT_DIR}"
echo "device=${DENSE_DEVICE}  allow_cpu_fallback=${ALLOW_CPU_FALLBACK}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"

module purge
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

APPTAINER_GPU_ARGS=()
if [[ "${NUM_GPUS}" -gt 0 ]]; then
  # Vega A100 nodes: --nv alone often fails to wire /dev/nvidia0 (only /dev/nvidia-caps shows up
  # inside the container), causing cudaErrorDevicesUnavailable on first CUDA op. See Sylabs/Apptainer
  # issue #523: --nvccli (libnvidia-container-cli backend) fixes device binding on MIG-capable A100
  # hosts. Set USE_NVCCLI=0 to fall back to plain --nv.
  USE_NVCCLI="${USE_NVCCLI:-1}"
  if [[ "${USE_NVCCLI}" == "1" ]]; then
    APPTAINER_GPU_ARGS+=(--nv --nvccli)
    echo "[gpu] Using --nv --nvccli (libnvidia-container-cli)"
  else
    APPTAINER_GPU_ARGS+=(--nv)
    echo "[gpu] Using --nv only (legacy)"
  fi
else
  echo "No GPUs allocated; MedEmbed build needs CUDA — request --gres=gpu:1"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

# Inner srun must request the GPU step on many Slurm sites (e.g. cgroup ConstrainDevices=yes):
# the sbatch allocation can include --gres=gpu:1 while a bare inner srun step still has no GPU
# devices in its cgroup, so the first real CUDA call fails with cudaErrorDevicesUnavailable.
#
# Important for Vega: keep Slurm's CUDA_VISIBLE_DEVICES in the container.
# With `singularity exec --cleanenv`, that variable is often dropped (observed as <unset>),
# which can break CUDA device mapping and produce cudaErrorDevicesUnavailable despite nvidia-smi.
# Therefore we avoid --cleanenv in this GPU job.

srun --mpi=none --gres=gpu:1 singularity exec \
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
    export SENTENCE_TRANSFORMERS_HOME=\"\$HF_HOME/sentence_transformers\"
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    mkdir -p \"\$HF_HOME\" \"\$HF_HUB_CACHE\" \"\$HF_HOME/transformers\" \"\$SENTENCE_TRANSFORMERS_HOME\"
    echo \"[cache] HF_HOME=\$HF_HOME\"

    export OMP_NUM_THREADS=8
    export PYTHONUNBUFFERED=1

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    echo \"[debug] SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-<unset>}  SLURM_STEP_GPUS=\${SLURM_STEP_GPUS:-<unset>}\"
    ls -l /dev/nvidia* || true
    nvidia-smi -L || true

    DEVICE='${DENSE_DEVICE}'
    if [[ \"\$DEVICE\" == \"cuda\" ]]; then
      echo \"[probe] validating CUDA runtime before model load\"
      if ! python - <<'PY'
import sys
import torch

print(f'[probe] torch={torch.__version__}')
print(f'[probe] cuda_available={torch.cuda.is_available()}')
print(f'[probe] device_count={torch.cuda.device_count()}')
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('CUDA unavailable in this job step')

try:
    x = torch.randn(1, device='cuda')
    y = x * 2
    del x, y
    torch.cuda.synchronize()
    print('[probe] CUDA allocation/synchronize OK')
except Exception as e:
    raise RuntimeError(f'CUDA probe failed: {e}')
PY
      then
        echo \"[probe] CUDA failed in this allocation (likely busy/unavailable GPU on node)\" >&2
        nvidia-smi || true
        if [[ '${ALLOW_CPU_FALLBACK}' == '1' ]]; then
          echo \"[probe] Falling back to CPU (ALLOW_CPU_FALLBACK=1)\" >&2
          DEVICE='cpu'
        else
          echo \"[probe] Set ALLOW_CPU_FALLBACK=1 to continue on CPU, or resubmit to get a healthy GPU allocation\" >&2
          exit 42
        fi
      fi
    fi

    python -u scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
      --jsonl_glob '${JSONL_GLOB}' \
      --out_dir '${OUT_DIR}' \
      --device \"\$DEVICE\" \
      --batch_size ${BATCH_SIZE} \
      --M ${M} \
      --ef_construction ${EF_CONSTRUCTION} \
      --ef_search ${EF_SEARCH} \
      --dedup_pmids
  "

echo "Finished job ${SLURM_JOB_ID:-local} at $(date)"

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
# article corpus (7c). 7c stores text under `abstract`, which the index
# builder reads via d.get("abstract").
#
# This script runs `singularity exec` directly from the sbatch shell (no
# inner srun) so Slurm's GPU env (CUDA_VISIBLE_DEVICES, /dev/nvidia*) is
# inherited cleanly. On Vega A100 nodes, `--nv` alone has been observed to
# leave /dev/nvidia0 / /dev/nvidiactl / /dev/nvidia-uvm out of the
# container, even though nvidia-smi works (it only needs /dev/nvidia-caps).
# We add explicit -B mounts for those device files when they exist on host.

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

JSONL_GLOB="/work/output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl"
OUT_DIR="/work/indexes/dicty_medembed_index_gpu"

BATCH_SIZE="${BATCH_SIZE:-256}"
M="${M:-32}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
EF_SEARCH="${EF_SEARCH:-100}"
DENSE_DEVICE="${DENSE_DEVICE:-cuda}"            # cuda|cpu
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-0}"   # 1 => fallback to CPU when CUDA probe fails
USE_NVCCLI="${USE_NVCCLI:-1}"                   # 1 => use --nvccli (libnvidia-container-cli, recommended on Vega)

echo "Starting job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Container: ${CONTAINER_IMG}"
echo "jsonl_glob=${JSONL_GLOB}  out_dir=${OUT_DIR}"
echo "device=${DENSE_DEVICE}  allow_cpu_fallback=${ALLOW_CPU_FALLBACK}  use_nvccli=${USE_NVCCLI}"

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

# ---------- Build singularity arg list ----------
# Strategy:
#   USE_NVCCLI=1 (default): use libnvidia-container-cli via --nvccli, with
#     NVIDIA_VISIBLE_DEVICES = the Slurm-allocated GPU id(s). nvccli handles
#     cgroup-correct device binding and avoids the NVML "Insufficient
#     Permissions" problem we hit when binding all /dev/nvidia[0-9]* by hand.
#   USE_NVCCLI=0: legacy --nv path. We then bind only the allocated /dev/nvidiaN
#     (read from SLURM_JOB_GPUS) plus auxiliary device files. Binding ALL GPU
#     devices breaks CUDA on cgroup-restricted multi-GPU nodes (NVML EPERM).
APPTAINER_ARGS=()

if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPUs allocated; MedEmbed build needs CUDA — request --gres=gpu:1"
fi

# Determine allocated GPU physical id(s) for --nvccli / smart binding.
# SLURM_JOB_GPUS is the comma-separated list of allocated GPU ids on this node.
ALLOC_GPU_IDS="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}"

if [[ "${NUM_GPUS}" -gt 0 && "${USE_NVCCLI}" == "1" ]]; then
  APPTAINER_ARGS+=(--nvccli)
  # nvccli uses NVIDIA_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES) to choose GPUs.
  export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${ALLOC_GPU_IDS:-all}"
  echo "[gpu] Using --nvccli  NVIDIA_VISIBLE_DEVICES=${APPTAINERENV_NVIDIA_VISIBLE_DEVICES}"
elif [[ "${NUM_GPUS}" -gt 0 ]]; then
  APPTAINER_ARGS+=(--nv)
  echo "[gpu] Using --nv (legacy)"

  # Auxiliary devices (always safe to bind when present on host).
  for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [[ -e "$d" ]] && APPTAINER_ARGS+=(-B "$d")
  done

  # Bind ONLY the allocated GPU device file(s). Binding all /dev/nvidiaN on a
  # cgroup-restricted multi-GPU node makes NVML hit EPERM on others and CUDA
  # init reports device_count=0.
  if [[ -n "${ALLOC_GPU_IDS}" ]]; then
    IFS=',' read -ra _ids <<< "${ALLOC_GPU_IDS}"
    for id in "${_ids[@]}"; do
      if [[ "$id" =~ ^[0-9]+$ && -e "/dev/nvidia${id}" ]]; then
        APPTAINER_ARGS+=(-B "/dev/nvidia${id}")
        echo "[gpu] Binding allocated /dev/nvidia${id}"
      fi
    done
  else
    echo "[gpu] No SLURM_JOB_GPUS detected; binding all /dev/nvidiaN (may fail under cgroup)"
    for d in /dev/nvidia[0-9]*; do
      [[ -e "$d" ]] && APPTAINER_ARGS+=(-B "$d")
    done
  fi
fi

APPTAINER_ARGS+=(
  -B "${WORKDIR}:/work"
  -B "${YUN_HOST}:/yun"
  -B "${HOME_HOST}:/home/wangy"
  --pwd /work
)

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-cache}"
mkdir -p "${APPTAINER_CACHEDIR}"

# Forward Slurm-provided GPU mapping into the container; avoid --cleanenv
# pitfall. Apptainer reads APPTAINERENV_* and exposes them inside the container.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

singularity exec \
  "${APPTAINER_ARGS[@]}" \
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

    echo \"[ctr] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    echo \"[ctr] /dev/nvidia* listing inside container:\"
    ls -l /dev/nvidia* 2>&1 || true
    nvidia-smi -L || true

    DEVICE='${DENSE_DEVICE}'
    if [[ \"\$DEVICE\" == \"cuda\" ]]; then
      echo \"[probe] validating CUDA runtime before model load\"
      if ! python - <<'PY'
import torch

print(f'[probe] torch={torch.__version__}')
print(f'[probe] cuda_available={torch.cuda.is_available()}')
print(f'[probe] device_count={torch.cuda.device_count()}')
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('CUDA unavailable in this job step')

x = torch.randn(1, device='cuda')
y = x * 2
del x, y
torch.cuda.synchronize()
print('[probe] CUDA allocation/synchronize OK')
PY
      then
        echo \"[probe] CUDA failed: device files present? check /dev/nvidia* listing above\" >&2
        nvidia-smi || true
        if [[ '${ALLOW_CPU_FALLBACK}' == '1' ]]; then
          echo \"[probe] Falling back to CPU (ALLOW_CPU_FALLBACK=1)\" >&2
          DEVICE='cpu'
        else
          echo \"[probe] Set ALLOW_CPU_FALLBACK=1 to run on CPU, or contact Vega support if /dev/nvidia0 is missing\" >&2
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

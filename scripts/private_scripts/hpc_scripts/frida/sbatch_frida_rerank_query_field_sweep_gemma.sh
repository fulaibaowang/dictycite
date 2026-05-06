#!/bin/bash
#SBATCH -J dicty_rerank_qf_7d_gemma
#SBATCH -p frida
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:A100:3
#
# Frida: rerank query-field sweep (7d) with BGE Gemma (LLM / FlagLLMReranker).
# Config: scripts/private_scripts/hpc_scripts/frida/config_frida_rerank_query_field_sweep_7d_gemma.env
# Matches GPU count with config (RERANK_NUM_GPUS=2).
#
# Host output root: /shared/workspace/biolab/yun/dicty_output/

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26b200.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"
DICTY_OUTPUT_HOST="/shared/workspace/biolab/yun/dicty_output"

PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/frida/config_frida_rerank_query_field_sweep_7d_gemma.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Rerank query-field sweep 7d (Gemma): ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"

mkdir -p "${DICTY_OUTPUT_HOST}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "[host] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# if [[ "${NUM_GPUS}" -lt 2 ]]; then
#   echo "Gemma rerank config expects 2 GPUs (RERANK_NUM_GPUS=2). Submit with --gres=gpu:A100:2." >&2
#   exit 1
# fi

srun \
  --container-image="${CONTAINER_IMG}" \
  --container-mount-home \
  --container-mounts "${WORKDIR}:/work,${PUBMED_HOST}:/pubmed,${DICTY_OUTPUT_HOST}:${DICTY_OUTPUT_HOST}" \
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

    echo \"[ctr] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    PIPELINE_CONFIG='${PIPELINE_CONFIG}'
    echo \"[sweep] Using config: \$PIPELINE_CONFIG\"
    set -a
    # shellcheck source=/dev/null
    source \"\$PIPELINE_CONFIG\"
    set +a
    mkdir -p \"\$WORKFLOW_SWEEP_OUTPUT_DIR\"
    cp \"\$PIPELINE_CONFIG\" \"\$WORKFLOW_SWEEP_OUTPUT_DIR/\"
    echo \"[run] Rerank query-field sweep (Gemma) -> \$WORKFLOW_SWEEP_OUTPUT_DIR\"
    ./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config \"\$PIPELINE_CONFIG\"

    echo \"[done] Rerank query-field sweep 7d (Gemma) completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

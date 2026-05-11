#!/bin/bash
#SBATCH -J dicty_rerank_qf_sweep
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# VEGA: rerank query-field sweep (same logic as query_expansion/sbatch_rerank_sweep.sh).
# Runs twice: gold benchmark 7d then 7e (7d minus evidence_level=abstract_insufficient).
# 1) Retrieval once per benchmark: BM25 + Dense both query_text_synonym_products, hybrid fusion.
# 2) Cross-encoder rerank three times: query_text, query_text_expansion_synonyms, query_text_synonym_products
#    (same fusion run TSVs each time; only rerank query text changes).
#
# Requires DOCS_JSONL, indexes, INPUT_JSONL in each config (config_vega_rerank_query_field_sweep_7d.env / _7e.env).
# Outputs per run: ${WORKFLOW_SWEEP_OUTPUT_DIR}/fixed_long_rerank_sweep/retrieval/{bm25,dense,fusion}/, rerank_{body,synonyms,long}/

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

SWEEP_CONFIGS=(
  scripts/private_scripts/hpc_scripts/vega/config_vega_rerank_query_field_sweep_7d.env
  scripts/private_scripts/hpc_scripts/vega/config_vega_rerank_query_field_sweep_7e.env
)

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Rerank query-field sweep (7d + 7e)"
printf '  %s\n' "${SWEEP_CONFIGS[@]}"
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
  echo "No GPUs allocated; rerank requires GPU. Run with --gres=gpu:1." >&2
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

    echo \"[debug] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    nvidia-smi -L || true

    for PIPELINE_CONFIG in \\
      scripts/private_scripts/hpc_scripts/vega/config_vega_rerank_query_field_sweep_7d.env \\
      scripts/private_scripts/hpc_scripts/vega/config_vega_rerank_query_field_sweep_7e.env; do
      echo \"[sweep] Using config: \$PIPELINE_CONFIG\"
      set -a
      # shellcheck source=/dev/null
      source \"\$PIPELINE_CONFIG\"
      set +a
      export WORKFLOW_SWEEP_OUTPUT_DIR=\"\${WORKFLOW_SWEEP_OUTPUT_DIR:-/home/wangy/dictycite_output/workflow_fixed_long_rerank_sweep}\"
      mkdir -p \"\$WORKFLOW_SWEEP_OUTPUT_DIR\"
      cp \"\$PIPELINE_CONFIG\" \"\$WORKFLOW_SWEEP_OUTPUT_DIR/\"
      echo \"[run] Rerank query-field sweep (retrieval long/long + rerank body/synonyms/long) -> \$WORKFLOW_SWEEP_OUTPUT_DIR\"
      ./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config \"\$PIPELINE_CONFIG\"
    done

    echo \"[done] Rerank query-field sweeps (7d + 7e) completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

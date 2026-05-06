#!/usr/bin/env bash
# Sourced from sbatch_vega_rerank_query_field_sweep_7d_gemma_{body,synonyms,long}.sh after cd to repo root.
# Requires: PIPELINE_CONFIG, RERANK_QF_ONLY in {body,synonyms,long}.

set -euo pipefail

: "${PIPELINE_CONFIG:?PIPELINE_CONFIG must be set}"
: "${RERANK_QF_ONLY:?RERANK_QF_ONLY must be set (body|synonyms|long)}"
case "${RERANK_QF_ONLY}" in
  body|synonyms|long) ;;
  *) echo "RERANK_QF_ONLY must be body, synonyms, or long (got: ${RERANK_QF_ONLY})" >&2; exit 1 ;;
esac

CONTAINER_IMG="/ceph/hpc/data/s25t12-03-users/apptainer/bioasq_08.03.26b200.sif"
WORKDIR="${PWD}"

PUBMED_HOST="/ceph/hpc/data/s25t12-03-users/pubmed"
YUN_HOST="/ceph/hpc/data/s25t12-03-users/"
HOME_HOST="/ceph/hpc/home/wangy"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Rerank query-field sweep 7d (Gemma), variant=${RERANK_QF_ONLY}: ${PIPELINE_CONFIG}"
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

    export RERANK_QF_ONLY='${RERANK_QF_ONLY}'
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

    PIPELINE_CONFIG='${PIPELINE_CONFIG}'
    echo \"[sweep] Using config: \$PIPELINE_CONFIG  RERANK_QF_ONLY=\$RERANK_QF_ONLY\"
    set -a
    # shellcheck source=/dev/null
    source \"\$PIPELINE_CONFIG\"
    set +a
    export WORKFLOW_SWEEP_OUTPUT_DIR=\"\${WORKFLOW_SWEEP_OUTPUT_DIR:-/home/wangy/dictycite_output/workflow_fixed_long_rerank_sweep}\"
    mkdir -p \"\$WORKFLOW_SWEEP_OUTPUT_DIR\"
    cp \"\$PIPELINE_CONFIG\" \"\$WORKFLOW_SWEEP_OUTPUT_DIR/\"
    echo \"[run] Rerank query-field sweep (Gemma, \$RERANK_QF_ONLY only) -> \$WORKFLOW_SWEEP_OUTPUT_DIR\"
    ./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config \"\$PIPELINE_CONFIG\"

    echo \"[done] Rerank query-field sweep 7d Gemma (\$RERANK_QF_ONLY) completed\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

#!/bin/bash
#SBATCH -J dicty_index_v2_chunked
#SBATCH -p frida
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:A100:1
#
# Builds the BM25 and dense (MedEmbed-HNSW) indexes for the v2 chunked
# full-text corpus. NO retrieval, NO rerank, NO eval — just the indexes.
#
# Corpus: dicty_fulltext_corpus/v2/corpus.jsonl
#         (10,880 abstracts + 87,028 body chunks from 3,556 PDFs)
# Indexes written to:
#   dicty_simulated_data/indexes/bm25_chunked_v2
#   dicty_simulated_data/indexes/medembed_chunked_v2
#
# After this job succeeds, point a downstream retrieval+rerank sbatch at
# config_frida_7a_public_goldset_chunked_fulltext_v2.env and it will pick
# up these prebuilt indexes via BM25_INDEX_PATH / DENSE_INDEX_DIR.

set -euo pipefail

cd ~/dictycite
mkdir -p logs

CONTAINER_IMG="/shared/home/yun.wang/biolab/yun/bioasq_08.03.26b200.sqfs"
WORKDIR="${PWD}"
PUBMED_HOST="/shared/workspace/biolab/pubmed"

PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/frida/config_frida_7a_public_goldset_chunked_fulltext_v2.env"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Config: ${PIPELINE_CONFIG}"
echo "Container image: ${CONTAINER_IMG}"

NUM_GPUS="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-0}}"
export NUM_GPUS
echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "[host] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[host] /dev/nvidia* listing:"
ls -l /dev/nvidia* 2>&1 || true
echo "[host] nvidia-smi -L:"
nvidia-smi -L 2>&1 || true

if [[ "${NUM_GPUS}" -eq 0 ]]; then
  export ENROOT_DISABLE_NVIDIA=1
  export NVIDIA_VISIBLE_DEVICES=void
  echo "No GPUs allocated; disabling NVIDIA hooks for container"
fi

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

    echo \"[ctr] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"
    echo \"[ctr] nvidia-smi -L:\"
    nvidia-smi -L || true

    echo \"[probe] validating CUDA runtime before indexing\"
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
      echo \"[probe] CUDA failed; aborting before indexing.\" >&2
      exit 42
    fi

    source '${PIPELINE_CONFIG}'

    CORPUS=\$DOCS_JSONL
    BM25_OUT=\$BM25_INDEX_PATH
    DENSE_OUT=\$DENSE_INDEX_DIR

    echo \"[indexes] CORPUS=\${CORPUS}\"
    echo \"[indexes] BM25_OUT=\${BM25_OUT}\"
    echo \"[indexes] DENSE_OUT=\${DENSE_OUT}\"

    if [ ! -f \"\${CORPUS}\" ]; then
      echo \"[error] Corpus not found: \${CORPUS}\" >&2
      exit 2
    fi

    mkdir -p \"\$(dirname \"\${BM25_OUT}\")\" \"\$(dirname \"\${DENSE_OUT}\")\"

    echo \"[step 1/2] Building BM25 (Terrier) index ...\"
    t0=\$(date +%s)
    python -m scripts.public.shared_scripts.index.build_bm25_index_from_jsonl_shards \\
      --jsonl_glob \"\${CORPUS}\" \\
      --index_path \"\${BM25_OUT}\" \\
      --threads 8 \\
      --overwrite
    t1=\$(date +%s)
    echo \"[timing] BM25 index: \$((t1-t0))s\"

    echo \"[step 2/2] Building dense (MedEmbed HNSW) index ...\"
    t0=\$(date +%s)
    python -m scripts.public.shared_scripts.index.build_dense_hnsw_index_from_jsonl_shards \\
      --jsonl_glob \"\${CORPUS}\" \\
      --out_dir \"\${DENSE_OUT}\" \\
      --model_name abhinand/MedEmbed-small-v0.1 \\
      --device cuda \\
      --batch_size 128 \\
      --max_seq_length 512
    t1=\$(date +%s)
    echo \"[timing] Dense index: \$((t1-t0))s\"

    echo \"[done] Indexes built\"
  "

echo "Finished job ${SLURM_JOB_ID} at $(date)"

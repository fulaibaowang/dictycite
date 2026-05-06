#!/bin/bash
#SBATCH -J dicty_rerank_qf_7d_gemma_body
#SBATCH -p gpu
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
#
# Vega: one Slurm job — rerank query-field sweep 7d (Gemma), query_text only (rerank_body).
# Shared config: config_vega_rerank_query_field_sweep_7d_gemma.env
# Submit synonyms + long separately; for parallel runs set RETRIEVAL_COPY_FROM in the config
# to a directory that already has .../fixed_long_rerank_sweep/retrieval/ (see config header).
#
# See also: sbatch_vega_rerank_query_field_sweep_7d_gemma_{synonyms,long}.sh

set -euo pipefail

cd ~/dictycite
mkdir -p logs

export PIPELINE_CONFIG="scripts/private_scripts/hpc_scripts/vega/config_vega_rerank_query_field_sweep_7d_gemma.env"
export RERANK_QF_ONLY=body
# shellcheck source=/dev/null
source "$(dirname "$0")/vega_rerank_qf_sweep_gemma_launch.sh"

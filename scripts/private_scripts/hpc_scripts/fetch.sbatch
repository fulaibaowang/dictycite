#!/bin/bash
#SBATCH -J dicty_epmc
#SBATCH -p frida
#SBATCH -c 2
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH -o article_fetching/output/slurm-%x-%A_%a.out
#SBATCH -e article_fetching/output/slurm-%x-%A_%a.err

set -euo pipefail

# Map array index -> date range
ranges=("1850-1979" "1980-1989" "1990-1999" "2000-2009" "2010-2019" "2020-2029")
range="${ranges[$SLURM_ARRAY_TASK_ID]}"
start="${range%-*}"
end="${range#*-}"

mkdir -p article_fetching/output

srun \
  --container-image=fulaibaowang/dictyfetch:19.01.2026 \
  --container-mount-home \
  --container-mounts "${PWD}/article_fetching/output:/app/output,${PWD}:/work" \
  --container-workdir /work \
  bash -lc "
    set -euo pipefail
    cd /dictycite/article_fetching
    BASE_OUTPUT=/app/output

    echo \"Running range: $start-$end\"
    python fetch.py \
      --query \"\\\"Dictyostelium\\\" AND FIRST_PDATE:[$start TO $end]\" \
      --output_path \"\$BASE_OUTPUT/$range\" \
      --get_text_from epmc_my \
      2>&1 | tee \"\$BASE_OUTPUT/$range.log\"
  "

#indexes
srun -p dev --time=12:00:00 --gres=gpu:1 -c 4 --mem=64G \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_20.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl"   --index_path "/work/indexes/dicty_bm25_index"   --threads 4   --overwrite

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/work/output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl" \
  --out_dir "/work/indexes/dicty_medembed_index" \
  --device "cuda" \
  --batch_size 256 \
  --M 32 \
  --ef_construction 200 \
  --ef_search 100 \
  --dedup_pmids

# pipeline (single run: one query-field for BM25, one for Dense)
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/hpc_scripts/config.env

# optional: set query-field overrides in config or env before running
# BM25_QUERY_FIELD=body_expansion_long   DENSE_QUERY_FIELD=body

# 3×3 query-field sweep (BM25 × Dense up to hybrid, no rerank)
# Writes 9 subdirs under WORKFLOW_OUTPUT_DIR: bm25_body_dense_body, bm25_body_dense_synonyms, ...
./scripts/private_scripts/hpc_scripts/query_expansion/run_query_field_sweep.sh --config scripts/private_scripts/hpc_scripts/config.env

# x3 query-field sweep (rerank)
./scripts/private_scripts/hpc_scripts/query_expansion/run_rerank_query_field_sweep.sh --config scripts/private_scripts/hpc_scripts/config.env

# Sweep output (sbatch): output/workflow_baseline_full_sweep/query_field_sweep/ (9 combos) and .../fixed_long_rerank_sweep/ (rerank). No overlap.
# Combine 3×3 results: python .../query_expansion/combine_query_field_sweep_results.py --workflow_dir output/workflow_baseline_full_sweep/query_field_sweep
#
# --- sbatch (fulltest config_full.env) ---
# Pipeline only (same as interactive pipeline with config_full.env):
#   cd ~/dictycite && sbatch scripts/private_scripts/hpc_scripts/fulltest/sbatch_pipeline_full.sh
# 3×3 BM25 × Dense query-field sweep (no rerank):
#   cd ~/dictycite && sbatch scripts/private_scripts/hpc_scripts/query_expansion/sbatch_query_field_sweep.sh
# Fixed long + rerank sweep (retrieval once + rerank body/synonyms/long):
#   cd ~/dictycite && sbatch scripts/private_scripts/hpc_scripts/query_expansion/sbatch_rerank_sweep.sh

# Combine 3×3 sweep (dense rows repeat by dense_query_field only — expected; figures are correct):
#   python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --workflow_dir output/workflow_baseline_full_sweep/query_field_sweep --log_x --plot bm25 dense --wide
# Combine rerank sweep:
#   python scripts/private_scripts/hpc_scripts/query_expansion/combine_query_field_sweep_results.py --rerank_sweep --workflow_dir output/workflow_baseline_full_sweep/fixed_long_rerank_sweep --log_x --wide

python scripts/public/shared_scripts/analysis/compare_result_dirs.py \
  --dirs output/workflow_baseline_full_sweep/fixed_long_rerank_sweep/rerank_body output/workflow_baseline_full_sweep/fixed_long_rerank_sweep/rerank_synonyms output/workflow_baseline_full_sweep/fixed_long_rerank_sweep/rerank_long output/workflow_baseline_full_sweep/fixed_long_rerank_sweep/hybrid \
  --labels "body" "synonyms" "long" "hybrid" \
  --plot both \
  --map-ks 1,5,10,20,30,50,100,200 \
  --train-json example/dicty_gold_llm_public_train_200.jsonl \
  --test-batch-jsons example/dicty_gold_llm_public_test_50.jsonl \
  --log-x --plots-by-split \
  --output-dir output/workflow_baseline_full_sweep/fixed_long_rerank_sweep/compare_plots
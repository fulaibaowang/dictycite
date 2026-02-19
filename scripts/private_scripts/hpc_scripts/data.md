#indexes
srun -p dev --time=12:00:00 --gres=gpu:1 -c 4 --mem=64G \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work" \
  --container-workdir /work \
  --pty bash

python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/cleaned/articles_all_cleaned_abstract.jsonl"   --index_path "/work/indexes/dicty_bm25_index"   --threads 4   --overwrite

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "/work/output/cleaned/articles_all_cleaned_abstract.jsonl" \
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
./scripts/private_scripts/hpc_scripts/run_query_field_sweep.sh --config scripts/private_scripts/hpc_scripts/config.env



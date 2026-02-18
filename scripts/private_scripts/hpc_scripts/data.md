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

# pipeline
srun -p dev --time=12:00:00 -c 4 --mem=64G --gres=gpu:1 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_04.02.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work" \
  --container-workdir /work \
  --pty bash

./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/hpc_scripts/config.env



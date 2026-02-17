#bm25
srun -p dev --time=12:00:00 -c 4 \
  --container-image=/shared/home/yun.wang/biolab/yun/bioasq_28.01.26.sqfs \
  --container-mount-home \
  --container-mounts "${PWD}:/work" \
  --container-workdir /work \
  --pty bash

python scripts/public/index/build_bm25_index_from_jsonl_shards.py   --jsonl_glob "/work/output/cleaned/articles_all_cleaned_abstract.json"   --index_path "/work/indexes/dicty_bm25_index"   --threads 4   --overwrite


#subset
python3 scripts/public/data_prep/make_goldset_subset.py \
  --input output/cleaned/dicty_gold_llm_public.json \
  --train-size 200 \
  --test-size 50 \
  --seed 42

#whole pipeline
./scripts/public/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/local/config.env --no-rerank

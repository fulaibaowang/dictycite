#subset
python3 scripts/public/data_prep/make_goldset_subset.py \
  --input output/cleaned/dicty_gold_llm_public.json \
  --train-size 200 \
  --test-size 50 \
  --seed 42

#whole pipeline
./scripts/public/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/local/config.env --no-rerank

#plots that splits different evidence_level 
python3 scripts/public/plot_by_evidence_level.py --workflow-dir output/workflow_hpc_test --gold-json output/cleaned/dicty_gold_llm_public.json
python3 scripts/public/plot_by_evidence_level.py --workflow-dir output/workflow_hpc_test --gold-json output/cleaned/dicty_gold_llm_public.json --rerank-dir rerank_bge

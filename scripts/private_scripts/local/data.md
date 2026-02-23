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

#plots for sweeping different query expansion
python scripts/public/combine_query_field_sweep_results.py \
  --workflow_dir output/workflow_hpc_test \
  --out output/workflow_hpc_test/combined_sweep_metrics.csv \
  --figures_dir output/workflow_hpc_test/figures

python scripts/public/combine_query_field_sweep_results.py \
  --workflow_dir output/workflow_hpc_test \
  --out output/workflow_hpc_test/combined_sweep_metrics.csv \
  --figures_dir output/workflow_hpc_test/figures   --log_x 

python scripts/public/combine_query_field_sweep_results.py --plot bm25 dense


python scripts/public/shared_scripts/compare_result_dirs.py \
  --dirs output/workflow_hpc_test/fixed_long_rerank_sweep/rerank_body output/workflow_hpc_test/fixed_long_rerank_sweep/rerank_synonyms output/workflow_hpc_test/fixed_long_rerank_sweep/rerank_long output/workflow_hpc_test/fixed_long_rerank_sweep/hybrid \
  --labels "body" "synonyms" "long" "hybird" \
  --plot both \
  --map-ks 10,20,50,100,200 \
  --train-json example/dicty_gold_llm_public_train_200.json \
  --test-batch-jsons example/dicty_gold_llm_public_test_50.json \
  --log-x \
  --output-dir output/workflow_hpc_test/fixed_long_rerank_sweep/compare_plots
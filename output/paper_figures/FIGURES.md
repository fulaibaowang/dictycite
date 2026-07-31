# Paper figures index

Consolidated location for all figures referenced in the manuscript and supplementary.
Generated PNGs are written to [`Figures/`](Figures/) (Overleaf-aligned path used by `paper.tex`).
Diagnostic / exploratory plots stay in their per-workflow `figures/` directories or `diagnostics/`.

## Prose and tables

- [`table1_ranker_comparison.md`](table1_ranker_comparison.md) — Table 1, ranker comparison on full goldset (n=1,656). Now carries the headline ranker result that was previously Fig 3 (`fig3_main_ranker_comparison.png`).
- Table 2 — qualitative claim/retrieval examples (`tab:qual_examples` in `paper.tex`); authored inline in the LaTeX source, no CSV here.
- [`table_s1_qe_reranker.md`](table_s1_qe_reranker.md) — Table S1, QE × reranker on QE subset (n=563).
- [`table_s2_evidence_haspdf_full.md`](table_s2_evidence_haspdf_full.md) — Table S2, evidence-level R@1000 and MRR@10 (BGE-m3) with inline Δ and 95% paired-bootstrap CIs in each +Chk cell, stacked as a has-PDF block (gains, MRR CIs above zero) and a no-PDF block (hurt by the added chunks, MRR CIs below zero). Replaces both the previous per-reranker Table S2 and Fig S3.

The prose for §3–§7 lives in the manuscript LaTeX source.

## Main figures

Manuscript figure numbers differ from the file numbers. `fig3_main_ranker_comparison.png` was dropped (its result is now Table 1), so the query-expansion and evidence figures appear as **Fig 3** and **Fig 4** in `paper.tex` even though their files are named `fig4_*` and `fig5_*`. The "Paper" column below is the number as it renders in the manuscript; the "File" column is the on-disk name.

| Paper | File | Source | Description |
|---|---|---|---|
| Fig 1 | `Figures/fig1_dataset2.png` | external (Inkscape / handmade) — not generated here | DictyBase curator-notes example with claim–citation pairs color-coded by evidence level (`abstract_supports_detail`, `abstract_supports_core`, `abstract_insufficient`). |
| Fig 2 | `Figures/fig2_pipeline_dicty.png` | external (Inkscape / handmade) — not generated here | Retrieval + reranking architecture: Query → BM25 + Dense → Retrieval fusion → Cross-encoder → Post-rerank fusion. |
| Fig 3 | `Figures/fig4_candidate_qe_main.png` | `notebooks/query_expansion_sweeping.py` | Query-expansion sweep main result — reranker MRR@K with and without query expansion. |
| Fig 4 | `Figures/fig5_evidence_level_retrieval_recall_rerank_mrr_haspdf.png` | `notebooks/report_7a.py` | Per-evidence-level (3 panels), has-PDF subset. **Upper row** = first-stage retrieval Recall@K (BM25+Dense hybrid fusion, K up to 5000). **Lower row** = post-rerank MRR@K (cross-encoder + post-rerank fusion). Two overlaid lines per panel: solid = abstract-only baseline, dashed = `+chunked corpus`. Story: chunked corpus rescues retrieval on `abstract_insufficient` (Recall ≈ 0.99 by K=2000) but reranking still bottlenecks (MRR@10 0.16 → 0.39, gap remains vs other buckets). |

**Generated but not in the current manuscript:** `Figures/fig3_main_ranker_comparison.png` (`notebooks/ragnarok_comparison.py`) — headline ranker comparison, MRR@K curves for our pipeline vs Ragnarok baselines. Replaced by Table 1; still produced for reference.

## Supplementary figures

| Paper | File | Source | Description |
|---|---|---|---|
| Fig S1 | `Figures/fig_s1_per_query_rerank_delta.png` | `notebooks/ragnarok_comparison.py` | Per-query MRR delta after reranking — distribution of gains/losses. |

**Generated but not in the current manuscript** (still produced for diagnostic use):

- `Figures/fig_unused_baseline_depths.png` (`notebooks/report_7a.py`) — two-stage pipeline illustration on the full goldset (n=1656). **Left** = Stage-1 mean Recall@K (BM25, Dense, BM25+Dense fusion) up to K=2000 — recall-focused first stage. **Right** = Stage-2 MRR@K (K≤50) for the fusion baseline vs BGE-m3 — ranking-focused second stage. Justifies the recall-then-rerank split: fusion already reaches ~0.97 recall by K=2000 while the reranker lifts MRR from ~0.57 to ~0.61. Was Fig S1 in an earlier draft; dropped for space, which is why the remaining supplement figure was renamed `fig_s1_*`.

- `fig_s3_evidence_level_full_goldset.png` (`notebooks/report_7a.py`) — full-dataset evidence-level retrieval/rerank. Table S2 now reports the has-PDF subset alongside its no-PDF complement instead of the full-dataset blend, saving a figure of space.
- `fig_s4_per_query_chunked_delta_haspdf.png` (`notebooks/report_7a.py` §15) — per-query Δ MRR@10 (chunked − abstract) by evidence level, BGE-m3, has-PDF subset (1×3 panels; per-bucket n = 707 / 585 / 350 across 1,480 unique queries). Helped/hurt/unchanged decomposition of the chunked-corpus gain. Helped fraction grows monotonically with evidence difficulty (15.7% / 34.7% / 46.9%); hurt fraction is non-trivial on the easier buckets (9.9% / 16.1%) but small on `abstract_insufficient` (8.0%). The PNG is no longer tracked in this directory.

## Conventions

- `figN_*.png` for main paper, `fig_sN_*.png` for supplementary (matches existing source-notebook conventions).
- Notebooks write PNGs to `output/paper_figures/Figures/` (paths used by `paper.tex` / Overleaf).
- All figures saved at `dpi=150, bbox_inches="tight"`.
- Source notebooks are paired `.ipynb` / `.py` via jupytext; edit either, run `jupytext --sync <file>` afterward.
- Underlying data:
  - Baseline (abstract-only) post-rerank fusion run: `output/workflow_vega_7a_public_goldset_both_routes/rerank/post_rerank_fusion_snippet/runs/best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR200_poolH200_k60.tsv`
  - `+chunked_v2` post-rerank fusion run: `output/workflow_frida_7a_public_goldset_chunked_v2/rerank/post_rerank_fusion_snippet/runs/best_rrf_7a_dicty_gold_llm_public_top5000_rrf_poolR200_poolH200_k60.tsv` (chunk-level docids; aggregated to PMID by max-pool, i.e. first-occurrence wins)
  - Evidence-level qrels: `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl` (`docs[].evidence_level`)
  - Has-PDF subset filter: `output/dicty_gold_build/7a_dicty_gold_pdf_coverage.tsv` (`in_chunks_v2 == "yes"`)

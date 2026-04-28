# data_prep

Scripts that build the gold dataset, in pipeline order. Steps 1–2 mirror notebooks `01_*` and `02_*`; downstream steps assemble, label, and export the final benchmark.

## Pipeline

| Script | Purpose | Outputs (default `output/dicty_gold_build/`) |
|---|---|---|
| `dicty_curator_notes.py` | Download DictyBase curator notes; extract claim sentences with citation anchors | `1_curator_notes.parquet`, `1_curator_claims.parquet`, `1_publications.parquet` |
| `dicty_publication.py` | Map DictyBase `publication_id` → PMID via gene References tabs | `2_gene_publication_pmid.parquet` |
| `dicty_claim_labeler.py` | Label (claim, citation) pairs via an Ollama-compatible LLM | JSONL with `doc_match`, `evidence_level`, `reason` |
| `build_gold_linked_notes_dataset.py` | Re-fetch raw notes for gold gene_ids; build canonical blocks + provenance | `8_raw_notes_snapshot.jsonl`, `8a–8d_*` |
| `make_goldset_subset.py` | Stratified train/test split of the public JSONL | Split JSONLs + stats JSON |
| `apply_query_expansion.py` | Append entity-aware expansion fields to query JSONL | JSONL with extra `query_text_expansion_*` fields |

All scripts are resume-safe — re-running skips already-processed keys.

Use `--help` on each script for arguments. Top-level usage examples are in `docs/USAGE.md`.

## Query expansion

`query_expansion/` is a standalone module — see [query_expansion/README.md](query_expansion/README.md). The operational config for this project is `conf/query_expansion_dicty_gene.yaml`.

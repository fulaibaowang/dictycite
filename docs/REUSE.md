# Reusing the abstract corpus and indexes

How to run your own retrieval over the paper's abstract-only corpus, starting from the prebuilt
indexes or rebuilding them. This covers the abstract corpus only. The full-text corpus is not
shared (see [FULLTEXT.md](FULLTEXT.md)).

## What there is

| Piece | Where | Notes |
|---|---|---|
| Corpus | `output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl` (in the repo; also on Zenodo) | 20,447 PubMed articles, title + abstract. This is the reranker's `DOCS_JSONL`. |
| Queries | `output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl` | Gold claims, field `query_text`. `7d` adds gene-expanded query fields. |
| BM25 index | `indexes/dicty_bm25_index/` (gitignored; ask the authors for a copy, or rebuild) | Terrier via PyTerrier, Porter stemmer + stopwords. |
| Dense index | `indexes/dicty_medembed_index/` (gitignored; same) | `abhinand/MedEmbed-small-v0.1`, 384-d, normalised, cosine HNSW (M=32, efC=200, efS=100), max 512 tokens, MeSH not included, PMIDs deduplicated. `meta.json` inside records all of it. |

The dense index only works with queries embedded by the same model. The pipeline reads the model
name from `meta.json`, so don't override it.

## Using the prebuilt indexes

1. Clone the repo and put the two index folders under `indexes/` in your checkout. The paper
   configs already point there (`$REPO_ROOT/indexes/dicty_bm25_index`,
   `$REPO_ROOT/indexes/dicty_medembed_index`). Copy them rather than symlinking: the job runs in a
   container, and a link to somewhere outside the mounted folders won't resolve.
2. Start from
   [`config_frida_fulltext_v3_7a_abs_bge.env`](../scripts/private_scripts/hpc_scripts/frida/config_frida_fulltext_v3_7a_abs_bge.env).
   It is the cleanest abstract-only config: RRF fusion fixed at k=60 with equal weights, which is
   what the paper used. Change these lines, which hold the authors' own paths:

   | Variable | Set to |
   |---|---|
   | `REPO` | your checkout |
   | `WORKSPACE` | a folder of yours for outputs (mounted as `/data` in the container) |
   | `OUT_DIR` | `/data/<run name>` |
   | `CONTAINER_IMG` | the pipeline container image (ask the authors, or build one from `scripts/public/shared_scripts/Dockerfile`) |
   | `RERANK_MODEL` | any cross-encoder; the paper compared `BAAI/bge-reranker-v2-m3` and `ncbi/MedCPT-Cross-Encoder` |

   Leave `REPO_ROOT=/work`. That is the checkout's path inside the container.
3. Submit with
   [`sbatch_pipeline_fulltext_v3.sh`](../scripts/private_scripts/hpc_scripts/frida/sbatch_pipeline_fulltext_v3.sh)
   (despite the name, it runs either corpus). It expects the authors' `hpc_submit.sh` wrapper. Plain
   `sbatch` works too:
   `PIPELINE_CONFIG=<your config> sbatch scripts/private_scripts/hpc_scripts/frida/sbatch_pipeline_fulltext_v3.sh`.
   Edit the `#SBATCH` partition and GPU lines for your cluster. The script also sets `HF_HOME` to a
   model cache under `/pubmed`; point it at your own cache, or set `HF_HUB_OFFLINE=1` if the cache
   is read-only for you.

What each pipeline parameter does:
[scripts/public/shared_scripts/README.md](../scripts/public/shared_scripts/README.md).

## Rebuilding the indexes

Both builds take minutes. Run them inside the pipeline container from the checkout root:

```bash
python scripts/public/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
  --index_path indexes/dicty_bm25_index --threads 16 --overwrite

python scripts/public/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob output/dicty_gold_build/7c_articles_cleaned_abstract.jsonl \
  --out_dir indexes/dicty_medembed_index --device cuda --batch_size 256 \
  --M 32 --ef_construction 200 --ef_search 100 --dedup_pmids
```

The model defaults to MedEmbed-small. A rebuild should match the shared indexes, but different
library versions can shift dense scores slightly.

# DictyCite - A Domain-Specific Biological Literature Retrieval Dataset for *Dictyostelium discoideum*

This release contains the curator-claim goldset, the EPMC abstract retrieval
corpus, and a query-expansion subset used in the paper:

> Benchmarking Domain-Specific Biological Literature Retrieval with *Dictyostelium*.
> Yun Wang et al. Manuscript in preparation / under review, 2026.

Code, pipeline configurations, and reproduction instructions:
<https://github.com/fulaibaowang/dictycite>

At a glance: **1,656** curator-claim queries spanning **862** unique
*Dictyostelium* genes, linked to **1,289** unique PubMed articles, evaluated
against a **20,447**-article EPMC abstract corpus.

---

## Files

- **`dictycite_goldset.jsonl`** - 1,656 queries.
  Curator-claim queries linked to cited PubMed articles, with LLM-assigned
  evidence-level labels.

- **`dictycite_abstract_corpus.jsonl`** - 20,447 articles.
  Cleaned PubMed abstracts forming the retrieval corpus.

- **`dictycite_query_expansion.jsonl`** - 563 queries.
  Subset of the goldset with gene-aware query-expansion variants
  (synonyms; synonyms and product descriptions).

- **`dictycite_fulltext_pmids.tsv`** - 1,124 PMIDs.
  List of gold PMIDs available in the chunked full-text variant, with PMC OA
  license tags (see *Full text* below).

Total uncompressed: ~40 MB.

---

## Schemas

### `dictycite_goldset.jsonl`

One JSON object per query. Top-level fields:

- `query_id` - string. Stable query identifier.
- `query_text` - string. Curator-written claim used as the retrieval query.
- `documents` - list of strings. PubMed URLs of cited (gold) articles.
- `docs` - list of objects. Cited articles with metadata (see below).
- `genes` - list of objects. dictyBase gene records linked to the query
  (`gene_id`, `gene_name`, `synonyms`, `gene_products`).
- `query_gene_expansion` - object. Gene-detection result
  (`has_detectable_gene`, `n_detected_genes`, `detected_gene_ids`,
  `query_expansion_benchmark`).

Each entry in `docs`:

- `pmid` - string. PubMed ID (joins to abstract corpus `pmid`).
- `publication_id` - int. dictyBase internal publication ID.
- `title` - string. Article title.
- `year` - int. Publication year.
- `abstract_clean` - string. Cleaned abstract text.
- `evidence_level` - string. LLM label, one of
  `abstract_supports_detail`, `abstract_supports_core`,
  or `abstract_insufficient`.
- `doc_match` - string. `yes` if the cited article was successfully
  matched to the abstract corpus.
- `reason` - string. LLM rationale for the evidence-level label
  (may be empty).
- `anchor_pos` - list. Character offsets of citation anchor positions
  in the source curator note.
- `citation_captions` - list. Raw citation strings extracted from
  the dictyBase page.

### `dictycite_abstract_corpus.jsonl`

One JSON object per article:

- `pmid` - string. PubMed ID (primary key; joins to goldset `docs[].pmid`).
- `docno` - string. Indexer document number (equal to `pmid`).
- `pmcid` - string. PMC ID, when available.
- `doi` - string. DOI, when available.
- `title` - string. Article title.
- `authors` - string. Comma-separated author list.
- `journal` - string. Journal name.
- `year` - string. Publication year.
- `text` - string. Cleaned abstract text.
- `type` - string. Always `"abstract"` in this release.
- `file` - string. Source file path (provenance only; not needed for use).

### `dictycite_query_expansion.jsonl`

Same schema as `dictycite_goldset.jsonl`, plus two expanded-query fields:

- `query_text_expansion_synonyms` - string. Original query + appended
  gene synonyms.
- `query_text_synonym_products` - string. Original query + appended
  gene synonyms and product descriptions.

Only queries with `query_expansion_benchmark == "yes"` are included.

### `dictycite_fulltext_pmids.tsv`

Tab-separated file. One row per gold PMID that has a chunked full-text
representation. Columns:

- `pmid` - PubMed ID (joins to goldset `docs[].pmid`).
- `pmcid` - PMC ID, empty when the article is not in PMC.
- `license` - PMC license tag (e.g., `CC BY`, `CC BY-NC`, `CC0`,
  `NO-CC CODE`) or `NON-OA` when the article is not in the PMC Open
  Access Subset.
- `n_chunks` - Number of full-text chunks for this PMID in our v2
  chunked corpus.

License distribution across the 1,124 PMIDs:
`NON-OA` 934, `CC BY` 86, `CC BY-NC-SA` 76, `NO-CC CODE` 13,
`CC BY-NC` 11, `CC0` 3, `CC BY-NC-ND` 1.

---

## Quick start

```python
import polars as pl
import json

# Load goldset and abstract corpus
queries = pl.read_ndjson("dictycite_goldset.jsonl")
corpus  = pl.read_ndjson("dictycite_abstract_corpus.jsonl")

# Iterate query / gold-doc pairs
with open("dictycite_goldset.jsonl") as f:
    for line in f:
        q = json.loads(line)
        for d in q["docs"]:
            print(q["query_id"], d["pmid"], d["evidence_level"])
            break
        break

# Join: goldset PMIDs to corpus abstracts
gold_long = (
    queries
    .select(["query_id", "query_text", "docs"])
    .explode("docs")
    .unnest("docs")
    .select(["query_id", "query_text", "pmid", "evidence_level"])
)
joined = gold_long.join(corpus.select(["pmid", "title", "text"]), on="pmid", how="left")
```

---

## Full text

The paper also reports results on a chunked full-text corpus variant. We do
**not** redistribute the chunked text here, because the underlying PDFs include
articles outside the PMC Open Access Subset that cannot be redistributed in
bulk. Instead, `dictycite_fulltext_pmids.tsv` lists the gold PMIDs that have a
chunked representation in our pipeline, together with PMC license tags so
readers can identify which articles are redistributable.

To reproduce the chunked corpus, see the PDF processing pipeline in the code
repository under `scripts/public/pdf_processing/`:

- `fetch_titles.py` - PMID -> PubMed title (NCBI esummary).
- `clean_pdfs.py` - PDFs -> cleaned body text + reference sections.
- `chunk_bodies.py` - bodies -> `chunks.jsonl`.

The full retrieval and reranking pipeline, including BM25, dense retrieval,
RRF fusion, and cross-encoder reranking, is under
`scripts/public/shared_scripts/` in the same repository.

---

## License

All data files in this release are licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Upstream attribution:

- **Curator claims and gene metadata** are derived from
  [dictyBase](http://dictybase.org/) gene summary pages. Please cite
  dictyBase when using the goldset.
- **Abstract text** is sourced from PubMed via Europe PMC. NLM does not
  claim copyright on PubMed abstracts, and abstracts may be subject to
  publisher or author copyright. Users are responsible for complying with the
  applicable
  [NLM/PubMed](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html)
  and publisher terms.

---

## Citation

```bibtex
@unpublished{dictycite2026,
  title  = {Benchmarking Domain-Specific Biological Literature Retrieval with Dictyostelium},
  author = {Wang, Yun and others},
  year   = {2026},
  note   = {Manuscript in preparation / under review. Dataset: [Zenodo DOI]; Code: \url{https://github.com/fulaibaowang/dictycite}}
}
```

---

## Funding

This work was funded by the European Union (Horizon Europe MSCA COFUND grant
No. 101081355 - SMASH).

## Acknowledgments

We thank dictyBase curators for the gene annotations and curator notes that
make this resource possible, and the National Library of Medicine / Europe PMC
for open access to PubMed metadata and abstracts.

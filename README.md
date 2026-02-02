# fetch data and inital cleaning

In order to prepare a training set of claim citation pair, I fetched
- [curator_notes on dictybase](#dicty_curator_notespy), these are curator notes (claims) with internal dictybase_publication_ids

|                                | Count |
|--------------------------------|-------|
| Curator notes (genes)          | 1,079 |
| Claims (from curator notes)    | 2,063 |


- [A mapping file of internal dictybase publication ids to pmids](#dicty_publicationpy), so that we can link the claims <-> dictybase_publication_ids <-> pmids. These info was not so straightforward to fetch from dictybase but luckily possible.

|                                               | Count |
|-----------------------------------------------|-------|
| All dictyBase publications with PMID          | 4,341 |
| Publications in curator notes with PMID       | 1,372 |


- [A full list of dicty literature in EPMC/PubMed](#article_fetchingfetchpy), from which we can get claims <-> pmids <-> titles/abstract

|                                                  | Count |
|--------------------------------------------------|-------|
| All Dictyostelium publications on EPMC           | 20,447 |
| Claims with titles/abstracts                     | 2,020 |
| Publications cited by these claims               | 1,340 |

Three sources were linked and cleaned up with **notebooks/dastasets.ipynb**, result in output file **output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet**

## dicty_curator_notes.py 

This script loop all genes over dictybase_files/DDB_G-curation_status.txt and fetch curator notes when existing.

Run a quick test on first 10
```
python dicty_curator_notes.py --limit 10
```

Run the full dataset
```
python dicty_curator_notes.py --limit 0 --sleep-base 0.3 --sleep-jitter 0.10
```

run with docker
```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 fulaibaowang/dictycite:16.01.2026 python dicty_curator_notes.py --limit 10
```

## dicty_publication.py

This script loop all genes over dictybase_files/DDB_G-curation_status.txt and fetch publications when existing. This is helpful to get publication ID <-> pmid pair.

run command similar as above
```
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 fulaibaowang/dictycite:16.01.2026 python dicty_publication.py --limit 10
```

##  article_fetching/fetch.py

Fetch literatures on EPMC. More instruction and docker usage in 
[article_fetching/README.md](article_fetching/README.md).

# gold datasets

With **notebooks/dastasets.ipynb** I curate the dataset furthur more.
- I cleanuped "()[];.."
- I grouped the claims into groups when pairwise TF-IDF value > 0.6.
- achieve ~1700 claims

Then I use BM25 method in **nBM25_query.ipynb** and eval the recall and MRR with bioASQ method. While we achieve a good recall@200 (see plots) with BM25 baseline, and our method of alias expansion worked, I can see this is clearly not a ground truth dataset because many claims are not fully supported (briefly mentioned) by abastract. Although BM25 method, a reranker might not furthur help.

To evaluate a reranker, I would do one of the followings:
- A. furthur curate gold claim dataset (with help of llm and good prompts)
  - A1: dicty_claim_labeler.py
  - A2: notebooks/goldset_llama.ipynb
- B. go with BioASQ dataset

## A1 dicty_claim_labeler.py

Further curated the gold claim–citation pairs using LLM-based labeling with **dicty_claim_labeler.py**:
- Labels each claim–citation pair with two judgments:
  - **doc_match**: whether the paper is the correct document-level citation for the claim (`yes`, `no`, `unclear`)
  - **evidence_level**: whether the abstract explicitly supports the core claim and/or details (`abstract_supports_detail`, `abstract_supports_core`, `needs_fulltext`, `not_applicable`)
- Designed structured prompt template 
- Input: `output/cleaned/gold_with_query_expand_flat.tsv`
- Output: `output/llm_labels_goldset_run<1|2|3>.jsonl`

Example usage:
```bash
python dicty_claim_labeler.py \
  --input_tsv output/cleaned/gold_with_query_expand_flat.tsv \
  --output_jsonl output/llm_labels_goldset_run3.jsonl \
  --key_path llama_API_KEY \
  --sleep_s 0.5 \
  --raise_on_error TRUE
```

## A2 notebooks/goldset_llama.ipynb
- example of llama approach
- checking agreement with three replicates
- customized logics to filter the gold set
  - doc_match=yes, evidence=abstract_supports_core | abstract_supports_detail

## B. go with BioASQ dataset
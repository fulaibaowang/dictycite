# dictycite

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

Three sources were linked and cleaned up with **dastasets.ipynb**, result in output file **output/cleaned/claim_cleaned_long_pmids_nonNA_abstract.parquet**

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
# dictycite

In order to prepare a training set of claim <-> citation pair, I fetched
- [curator_notes on dictybase](#dicty_curator_notespy), these are curator notes (claims) with internal dictybase publication ids

|                                | Count |
|--------------------------------|-------|
| Curator notes (genes)          | 1,079 |
| Claims (from curator notes)    | 2,677 |


- [A mapping file of internal dictybase publication ids to pmids](#dicty_publicationpy), so that we can link the claims to pmids. These info was not so straightforward to fetch from dictybase but luckily possible.

All publications on dictybase with pmid 4341
publications in curator notes 1410
publications in curator notes that not mapped (no pmid) 29

- [A full list of dicty literature in EPMC/PubMed](#article_fetchingfetchpy)
All Dictyostelium publications 
With abstract
With abstract and full text
Overlapped with



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
# dictycite

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
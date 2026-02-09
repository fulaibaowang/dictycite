# DictyCite: data fetching + RAG dataset prep

This repo collects data sources and preprocessing steps for building a Dictyostelium RAG dataset.
It keeps scripts, notebooks, and outputs separate and ready for public release.

## Methods
- get query-citation pair from Dictybase
  - get curators notes
  - clean up 
- prepare literature corpus
- retreival

For full workflows and Docker/HPC recipes, see [docs/USAGE.md](docs/USAGE.md).

## Results

Brief stats and dataset notes live in [docs/RESULTS.md](docs/RESULTS.md).


## Quickstart

Generate curator notes (sample):
```
python scripts/public/dicty_curator_notes.py --limit 10
```

Generate publication ID to PMID map (sample):
```
python scripts/public/dicty_publication.py --limit 10
```


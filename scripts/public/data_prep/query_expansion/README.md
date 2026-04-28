# query_expansion

Entity-aware query expansion for BM25 retrieval. Given a query string and a tabular entity
database, it detects entity mentions and appends their aliases and descriptions as structured
suffixes — helping BM25 match synonym variants it would otherwise miss.

## Quickstart

```bash
pip install pyyaml polars
python apply_query_expansion.py \
  --config conf/example.yaml \
  --table path/to/entities.tsv \
  --input queries.jsonl \
  --output queries_expanded.jsonl
```

Input JSONL must have one JSON object per line with a query field (default `"query"`).
Each output record gets one extra field per expansion variant (names defined in the YAML).

## How it works

**Detection**: the query is tokenized and each token is matched against an alias lookup
built from the entity table. A detected entity is identified by the token that triggered it.

**Expansion**: detected entities get a structured suffix appended:

```
<query text>
<token>: <alias1>, <alias2>, <description>
```

Multiple entities are joined with ` ||| `. The suffix is only appended when it adds terms
not already in the query.

**Tiering** (controls how much gets appended based on how many entities are detected):

| Detected | Behaviour |
|---|---|
| 1–2 | Full: name + filtered aliases (+ description if variant includes it) |
| 3 | Light: canonical name only (+ description for non-strict variants) |
| 4+ | Minimal: first entity, name only |

## YAML config

See `conf/example.yaml` for a fully annotated template. Key fields:

```yaml
entity_id_col: "EntityID"      # column that uniquely identifies each entity

detect_from:                   # columns whose values are tokenised and matched in queries
  - "Name"
  - "Aliases"

comma_split_cols:              # columns that hold comma-separated lists of aliases
  - "Aliases"

expand_variants:
  names_only:
    output_field: "query_expanded_names"   # field added to each output record
    expand_with:                           # columns used to build the expansion suffix
      - "Name"
      - "Aliases"
    exclude_from_output:                   # never append these to the suffix
      - "EntityID"

  names_and_description:
    output_field: "query_expanded_full"
    expand_with:
      - "Name"
      - "Aliases"
      - "Description"           # not in detect_from → treated as description: appended whole
    exclude_from_output:
      - "EntityID"

filters:
  min_alias_len: 3
  max_aliases_per_entity: 3
  auto_block_if_seen_in_n_entities: 5
  blocklist: []
```

### The detect_from / expand_with distinction

- **`detect_from`** controls which columns contribute aliases to the entity lookup used for
  detection. Tokens in query text are matched against this alias map.
- **`expand_with`** per variant controls which columns appear in the expansion suffix.
- A column in `expand_with` but **not** in `detect_from` is treated as a description:
  its value is appended whole (not comma-split, not filtered) after the alias tokens.
  This is how you include a free-text description column without polluting the alias lookup.

### Alias filters

All applied during index construction (before any query is processed):

| Filter | Effect |
|---|---|
| `min_alias_len` | Drop aliases shorter than N characters |
| `max_aliases_per_entity` | Keep at most N aliases per entity in the expansion |
| `auto_block_if_seen_in_n_entities` | Drop aliases that appear across N or more distinct entities (too ambiguous) |
| `blocklist` | Always drop these exact strings (lowercase) |

## Python API

```python
from pathlib import Path
from query_expansion.config import load_expansion_config
from query_expansion.table import build_entity_index
from query_expansion.expand import detect_genes, expand_query_structured

cfg = load_expansion_config(Path("conf/my_config.yaml"))
index = build_entity_index(Path("entities.tsv"), cfg)

# Detection only
detected = detect_genes("my query text", index)
# → {"ENTITY_ID": "matched_token", ...}

# Expansion
expanded, detected_ids, suffix = expand_query_structured("my query text", index, "names_only")
# → (full expanded string, [sorted entity ids], suffix string)
```

## DictyBase config

`conf/query_expansion_dicty_gene.yaml` is the operational config for this project.
It uses `dictybase_files/gene_information.txt` as the entity table (columns: GENE ID,
Gene Name, Synonyms, Gene products). The `long` variant appends gene products in addition
to synonyms; the `synonyms_only` variant omits them.

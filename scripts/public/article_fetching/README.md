# article_fetching

Fetch article metadata and (optionally) full text from Europe PMC / NCBI. Each article is saved as `<PMID>.json`.

## fetch.py

```bash
python fetch.py --query 'OPEN_ACCESS:y AND "Dictyostelium discoideum"' \
                --get_text_from ncbi_my \
                --output_path results
```

| Flag | Default | Notes |
|---|---|---|
| `--query` | required | Europe PMC [query syntax](https://europepmc.org/Help#query-syntax) |
| `--max_records` | all matches | |
| `--get_text_from` | `None` (metadata only) | `epmc`, `epmc_my`, `ncbi`, `ncbi_my` — see below |
| `--output_path` | timestamped dir | |

### Output JSON

```json
{
  "id": "...", "pmid": "...", "pmcid": "...", "url": "...",
  "title": "...", "authors": "...", "journal": "...", "year": "...",
  "doi": "...", "license": "CC BY", "abstract": "...",
  "text": { "Title": [...], "Abstract": [...], "Introduction": [...], ... }
}
```

`text` is `null` unless `--get_text_from` is set. Articles without a PMCID generally have no full text in PMC.

### Text fetching backends

| Value | Source | Implementation |
|---|---|---|
| `epmc` | Europe PMC fullTextXML | R `tidypmc` (requires R) |
| `epmc_my` | Europe PMC fullTextXML | Pure Python (JATS parser) |
| `ncbi` | NCBI BioC API | Section-tagged passages |
| `ncbi_my` | NCBI PMC efetch | Pure Python (JATS parser) |

Results are roughly equivalent; `ncbi` is the recommended default.

## utils

- `utils/filter_by_license.py --input_path DIR [--output_path DIR_filtered]` — drop articles with `CC BY-ND`, `CC BY-NC-ND`, or non-CC licenses.
- `utils/analyse.py --input_path DIR` — print full-text coverage and license distribution.

## Docker

```bash
docker run --rm \
  -v "$(pwd)/scripts/public/article_fetching/output:/app/output" \
  fulaibaowang/dictyfetch:19.01.2026 \
  --query 'OPEN_ACCESS:y AND "Dictyostelium discoideum"' \
  --output_path /app/output \
  --get_text_from epmc_my \
  --max_records 2
```

# Article Fetching

## Overview

This module provides tools for fetching scientific article metadata and full texts from Europe PMC (which aggregates PubMed and PubMed Central). The main script `fetch.py` retrieves article information and optionally downloads full text content, saving everything as structured JSON files.

### Key Concepts

**PubMed vs PubMed Central (PMC):**
- **PubMed** is a bibliographic index that stores article metadata (title, authors, abstract, journal, DOI, etc.)
- **PubMed Central (PMC)** is a full-text repository that stores complete article content

**Identifiers:**
- **PMID** (PubMed ID): Unique identifier for articles in PubMed
- **PMCID** (PubMed Central ID): Unique identifier for articles in PMC
  - If an article has no PMCID, the full text is typically not available in PMC

## Main Script: `fetch.py`

### Description

Fetches article metadata from Europe PMC API and optionally downloads full text content from various sources. Each article is saved as a separate JSON file named `<PMID>.json`.

### Usage

```bash
python fetch.py --query "YOUR_SEARCH_QUERY" [OPTIONS]
```

### Arguments

- `--query` (required): Search query string for Europe PMC API
  - Example: `TITLE:"Dictyostelium discoideum" AND TITLE:aggregation`
  - See [Europe PMC query syntax](https://europepmc.org/Help#query-syntax) for details

- `--max_records` (optional): Maximum number of records to fetch
  - Default: `None` (fetches all matching records)
  - Example: `--max_records 100`

- `--get_text_from` (optional): Source for full text fetching
  - Options: `"epmc"`, `"epmc_my"`, `"ncbi"`, `"ncbi_my"`, or `None`
  - Default: `None` (no full text fetched)
  - See [Text Fetching Methods](#text-fetching-methods) below

- `--output_path` (optional): Directory where JSON files will be saved
  - Default: Creates a timestamped directory (e.g., `fetch-2025-01-18_14-27-54`)
  - Example: `--output_path results/my_fetch`

### Example

```bash
# Fetch all open-access articles about Dictyostelium discoideum
python fetch.py \
    --query 'OPEN_ACCESS:y AND "Dictyostelium discoideum"' \
    --get_text_from ncbi
```

### Output Format

Each JSON file contains the following fields:

```json
{
  "id": "Europe PMC ID",
  "pmid": "PubMed ID",
  "pmcid": "PubMed Central ID (may be null)",
  "url": "https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
  "title": "Article title",
  "authors": "Author string",
  "journal": "Journal name",
  "year": "Publication year",
  "doi": "DOI (may be null)",
  "license": "License type (e.g., 'CC BY', 'CC BY-NC')",
  "citation": {
    "apa": "Full APA citation",
    "apa_short": "Short APA citation"
  },
  "abstract": "Abstract text",
  "text": null  // Full text dict if --get_text_from was specified
}
```

The `text` field (when present) is a dictionary with section names as keys and lists of paragraphs as values:
```json
{
  "Title": ["Article Title"],
  "Abstract": ["Paragraph 1", "Paragraph 2", ...],
  "Introduction": ["Paragraph 1", ...],
  "Methods": ["Paragraph 1", ...],
  ...
}
```

## Text Fetching Methods

The `--get_text_from` parameter supports four different methods for fetching full text:

1. **`epmc`**: Uses R package `tidypmc` to parse Europe PMC XML
   - Requires R and the `tidypmc` R package
   - Returns structured text by section

2. **`epmc_my`**: Custom Python parser for Europe PMC fullTextXML
   - Pure Python implementation
   - Parses JATS XML format
   - Returns structured text by section

3. **`ncbi`**: Fetches from NCBI BioC API
   - Returns passages with section types
   - Different structure than other methods

4. **`ncbi_my`**: Custom Python parser for NCBI PMC efetch
   - Uses NCBI eutils API
   - Parses JATS XML format
   - Returns structured text by section

**Recommendation:** Use `ncbi` for best results. Other three methods give more-or-less similar results.

## Utility Scripts

### 1. `utils/filter_by_license.py`

Filters JSON files based on license type and PMCID availability.

**Usage:**
```bash
python utils/filter_by_license.py \
    --input_path ... \
    --output_path ..._filtered
```

**Filters:**
- Removes articles with invalid licenses: `CC BY-ND`, `CC BY-NC-ND`, or `CC`

**Arguments:**
- `--input_path` (required): Source directory containing JSON files
- `--output_path` (optional): Destination directory (default: `{input_path}_filtered`)

### 2. `utils/analyse.py`

Analyzes JSON files to provide statistics about text content and license distribution.

**Usage:**
```bash
python utils/analyse.py --input_path results/dicty_aggregation_filtered
```

**Output:**
- Total articles processed
- Percentage of articles with full text
- Text content analysis (Title only, Abstract only, etc.)
- License distribution statistics

**Arguments:**
- `--input_path` (required): Directory containing JSON files to analyze

### 3. `utils/save_text.py`

Extracts abstracts (or other text) from JSON files and saves them as individual text files.

**Usage:**
```bash
python utils/save_text.py \
    --input_path results/dicty_aggregation_filtered \
    --output_path results/dicty_aggregation_text \
    --scope abstracts
```

**Arguments:**
- `--input_path` (required): Directory containing JSON files
- `--output_path` (required): Directory where text files will be saved
- `--scope` (optional): Output format - currently only `"abstracts"` is supported (default: `"abstracts"`)

## License Types

When filtering articles, the following license types are considered:

- **`CC0`** (Public Domain Dedication): No restrictions
- **`CC BY`** (Creative Commons Attribution): Must give credit
- **`CC BY-SA`**: Same as CC BY, must share under same license
- **`CC BY-NC`**: Same as CC BY, non-commercial use only
- **`CC BY-NC-SA`**: Combination of CC BY-NC and CC BY-SA
- **`CC BY-ND`**: Same as CC BY, no derivatives allowed (filtered out)
- **`CC BY-NC-ND`**: Same as CC BY-ND, non-commercial (filtered out)
- **`non-CC`**: Publisher-specific licenses (filtered out)

**Note:** The `OPEN_ACCESS:y` filter in queries does not guarantee Creative Commons licensing. Always check the `license` field in the output.

## Workflow Example

A typical workflow for fetching and processing articles:

```bash
# Step 1: Fetch articles
query='OPEN_ACCESS:y AND "Dictyostelium discoideum"'
python fetch.py \
    --query "$query" \
    --get_text_from epmc_my \
    --output_path results

# Step 2: Filter by license and PMCID
python utils/filter_by_license.py \
    --input_path results

# Step 3: Analyze results
python utils/analyse.py \
    --input_path results_filtered

# Step 4: Extract abstracts (optional)
python utils/save_text.py \
    --input_path results/dicty_aggregation_filtered \
    --output_path texts
```

## Historical Data

### 2025-07-18_14-27-54

A fetch was performed for all open-access articles on *Dictyostelium discoideum*:
- Query: `OPEN_ACCESS:y AND "Dictyostelium discoideum"`
- Total articles found: **5,259**
- After filtering (license + PMCID + text extraction): **4,590 articles**

Analysis revealed:
- 93.38% of articles had more than just Title/Abstract
- License distribution:
  - CC BY: 76.88%
  - CC BY-NC-SA: 13.44%
  - CC BY-NC: 9.00%
  - CC0: 0.68%

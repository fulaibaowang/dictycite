#!/usr/bin/env python3
"""
Download dictyBase References (publication_id + PMID) for a list of gene IDs,
but only when the gene page declares a References tab.

Steps per gene:
  1) GET BASE + "/gene/{gene_id}" and parse embedded `var config = [...]`
  2) If References tab exists, fetch its `source` (usually "/gene/{gene_id}/references.json")
  3) Parse rows from references.json table records:
       gene_id, publication_id, pmid

Output (resumable):
  - gene_id, publication_id, pmid
  - default: output/dicty_gold_build/2_gene_publication_pmid.parquet

Resume behavior:
  - Resume keyed off gene_id: once a gene_id appears in output, it’s skipped on later runs.
  - Genes with no References tab are recorded as a marker row:
      (gene_id, None, None)

Optional gene list behavior:
  - If --gene-list PATH is provided, only those gene IDs (one per line) are processed.
    (Blank lines and lines starting with # are ignored.)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import polars as pl
import requests

# Keep BASE unchanged
BASE = "http://dictybase.org"

PUB_RE = re.compile(r"/publication/(\d+)")
PMID_RE = re.compile(
    r"(?:/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)|view\.ncbi\.nlm\.nih\.gov/pubmed/(\d+)"
)

# Extract the JSON array assigned to `var config = [...]` before `var panel`
CONFIG_RE = re.compile(r"var\s+config\s*=\s*(\[\{.*?\}\]);\s*var\s+panel", re.DOTALL)

# ---------------------------
# Session / politeness helpers
# ---------------------------

def make_session() -> requests.Session:
    """Create a requests session with a friendly User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": "dictybase-references/0.2"})
    return session

def polite_sleep(base: float = 0.15, jitter: float = 0.10) -> None:
    """Small delay to avoid hammering the server."""
    time.sleep(base + random.random() * jitter)

def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: float,
    max_retries: int,
    retry_backoff_base: float,
    headers: Optional[dict] = None,
) -> requests.Response:
    """GET with exponential backoff retries. Raises after exhausting retries."""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, timeout=timeout, headers=headers)
            return r
        except requests.RequestException as e:
            last_err = e
            if attempt >= max_retries:
                raise
            backoff = retry_backoff_base * (2 ** attempt) + random.random() * 0.2
            print(f"GET failed ({url}): {e}; retrying in {backoff:.2f}s")
            time.sleep(backoff)
    # should not reach
    raise RuntimeError(last_err)

# ---------------------------
# Input / resume / output I/O
# ---------------------------

def load_genes_status() -> pl.DataFrame:
    """
    Load gene IDs from dictyBase status file.

    IMPORTANT: keep this snippet as requested.
    """
    df_status = pl.read_csv(
        "dictybase_files/DDB_G-curation_status.txt",
        separator="\t",
        has_header=False,
        truncate_ragged_lines=True,
    )
    genes_status = df_status.select(df_status.columns[0]).unique()
    return genes_status

def load_gene_list(path: str) -> pl.Series:
    """Load gene IDs from a text file (one per line). Lines starting with # are ignored."""
    ids: List[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return pl.Series(ids, dtype=pl.Utf8)

def load_done_ids(out_path: Path) -> set[str]:
    """If output exists, load processed gene IDs to enable resume."""
    if not out_path.exists():
        return set()
    df_done = pl.read_parquet(out_path)
    if "gene_id" not in df_done.columns:
        return set()
    return set(df_done["gene_id"].drop_nulls().unique().to_list())

def append_parquet(
    out_path: Path,
    rows: List[Dict[str, Any]],
    unique_subset: Optional[List[str]] = None,
) -> None:
    """
    Append a batch of rows to Parquet (or create the file if absent).
    If unique_subset is provided, deduplicate on those columns after concatenation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    df_batch = pl.DataFrame(rows).with_columns(
        pl.col("gene_id").cast(pl.Utf8),
        pl.col("publication_id").cast(pl.Utf8),
        pl.col("pmid").cast(pl.Utf8),
    )

    if out_path.exists():
        try:
            df_existing = pl.read_parquet(out_path)
            df_combined = pl.concat([df_existing, df_batch], how="vertical")
            if unique_subset:
                df_combined = df_combined.unique(subset=unique_subset, keep="first")
            df_combined.write_parquet(out_path)
        except Exception as e:
            print(f"Warning: failed to append to {out_path} ({e}), writing batch only")
            df_batch.write_parquet(out_path)
    else:
        df_batch.write_parquet(out_path)

# ---------------------------
# Gene page -> references source
# ---------------------------

def get_references_source_from_gene_page_html(html: str) -> str | None:
    """
    Given gene page HTML, return the 'source' URL for references.json if the References tab exists.
    Otherwise return None.
    """
    m = CONFIG_RE.search(html)
    if not m:
        return None

    config_json = m.group(1)
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        return None

    try:
        tabs = config[0]["items"]
    except Exception:
        return None

    for tab in tabs:
        if not isinstance(tab, dict):
            continue

        if tab.get("key") == "references":
            src = tab.get("source")
            return src if isinstance(src, str) and src else None

        src = tab.get("source")
        if isinstance(src, str) and src.endswith("/references.json"):
            return src

    return None

def get_references_source_for_gene(
    gene_id: str,
    session: requests.Session,
    timeout: float,
    max_retries: int,
    retry_backoff_base: float,
) -> str | None:
    """Fetch gene page and detect references source path if present."""
    url = f"{BASE}/gene/{gene_id}"
    r = request_with_retries(
        session=session,
        url=url,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_base=retry_backoff_base,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return get_references_source_from_gene_page_html(r.text)

# ---------------------------
# references.json parsing
# ---------------------------

def _find_records_lists(node: Any) -> Iterable[list]:
    """Yield every list found at any nested key named 'records'."""
    if isinstance(node, dict):
        recs = node.get("records")
        if isinstance(recs, list):
            yield recs
        for v in node.values():
            yield from _find_records_lists(v)
    elif isinstance(node, list):
        for v in node:
            yield from _find_records_lists(v)

def _clean_url(u: str) -> str:
    """Trim duplicated concatenated URLs (rare)."""
    if not u:
        return u
    first = u.find("http")
    second = u.find("http", first + 4)
    if first == 0 and second > 0:
        return u[:second]
    return u

def extract_pubid_pmid_from_references_json(
    gene_id: str,
    payload: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str | None, str | None]] = set()

    for records in _find_records_lists(payload):
        for rec in records:
            if not isinstance(rec, dict):
                continue

            ref_links = rec.get("ref_link") or []
            if not isinstance(ref_links, list):
                continue

            pub_id: str | None = None
            pmid: str | None = None

            for it in ref_links:
                if not isinstance(it, dict):
                    continue
                u = _clean_url((it.get("url") or "").strip())
                if not u:
                    continue

                m = PUB_RE.search(u)
                if m:
                    pub_id = m.group(1)

                m = PMID_RE.search(u)
                if m:
                    pmid = m.group(1) or m.group(2)

            if not pub_id and not pmid:
                continue

            key = (pub_id, pmid)
            if key in seen:
                continue
            seen.add(key)

            rows.append({"gene_id": gene_id, "publication_id": pub_id, "pmid": pmid})

    return rows

def fetch_references_rows(
    gene_id: str,
    ref_source: str,
    session: requests.Session,
    timeout: float,
    max_retries: int,
    retry_backoff_base: float,
) -> List[Dict[str, Any]]:
    """Fetch the references.json payload from source path and parse rows."""
    ref_url = ref_source if ref_source.startswith("http") else (BASE + ref_source)

    r = request_with_retries(
        session=session,
        url=ref_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_base=retry_backoff_base,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    if r.status_code == 404:
        return []
    r.raise_for_status()

    payload = r.json()
    return extract_pubid_pmid_from_references_json(gene_id, payload)

# ---------------------------
# Main runner (batch + resume)
# ---------------------------

def run(
    out_path: Path,
    limit: int | None,
    batch_size: int,
    timeout: float,
    sleep_base: float,
    sleep_jitter: float,
    max_retries: int,
    retry_backoff_base: float,
    gene_list_path: str | None,
) -> None:
    session = make_session()

    if gene_list_path:
        gene_series = load_gene_list(gene_list_path)
    else:
        genes_status = load_genes_status()
        gene_series = genes_status.to_series()

    if limit is not None:
        gene_series = gene_series.head(limit)

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} genes already processed in {out_path}")
    else:
        print("Starting fresh")

    buffer: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for gid in gene_series:
        gid = str(gid)

        if gid in done_ids:
            skipped += 1
            continue

        # 1) check if references tab exists and where it points
        ref_source = None
        try:
            ref_source = get_references_source_for_gene(
                gene_id=gid,
                session=session,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff_base=retry_backoff_base,
            )
        except Exception as e:
            print(f"{gid}: failed to fetch/parse gene page ({e})")

        if not ref_source:
            # No references tab -> record marker row so resume works
            buffer.append({"gene_id": gid, "publication_id": None, "pmid": None})
            processed += 1
        else:
            # 2) fetch references.json and parse
            try:
                rows = fetch_references_rows(
                    gene_id=gid,
                    ref_source=ref_source,
                    session=session,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_backoff_base=retry_backoff_base,
                )
                if rows:
                    buffer.extend(rows)
                else:
                    # references tab exists but JSON missing/empty
                    buffer.append({"gene_id": gid, "publication_id": None, "pmid": None})
                processed += 1
            except Exception as e:
                print(f"{gid}: failed to fetch/parse references.json ({e})")
                # mark as done (optional); keeps resume consistent
                buffer.append({"gene_id": gid, "publication_id": None, "pmid": None})
                processed += 1

        if len(buffer) >= batch_size:
            append_parquet(out_path, buffer, unique_subset=["gene_id", "publication_id", "pmid"])
            buffer.clear()
            done_ids = load_done_ids(out_path)  # refresh
            print(f"Saved progress. New genes: {processed} (skipped {skipped})")

        polite_sleep(base=sleep_base, jitter=sleep_jitter)

    if buffer:
        append_parquet(out_path, buffer, unique_subset=["gene_id", "publication_id", "pmid"])
        print(f"Saved final batch. Total new genes: {processed} (skipped {skipped})")

    print("Done.")
    print(f"Output: {out_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape dictyBase references via References tab detection into Parquet.")
    p.add_argument(
        "--out",
        default="output/dicty_gold_build/2_gene_publication_pmid.parquet",
        help="Output parquet path (default: output/dicty_gold_build/2_gene_publication_pmid.parquet)",
    )
    p.add_argument("--limit", type=int, default=10, help="How many genes to process (default: 10). Use 0 for all.")
    p.add_argument("--batch-size", type=int, default=5000, help="Write every N rows (default: 5000).")
    p.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default: 15).")
    p.add_argument("--sleep-base", type=float, default=0.15, help="Base sleep between requests in seconds (default: 0.15).")
    p.add_argument("--sleep-jitter", type=float, default=0.10, help="Additional random jitter in seconds (default: 0.10).")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per request (default: 3).")
    p.add_argument("--retry-backoff-base", type=float, default=0.8, help="Backoff base seconds (default: 0.8).")
    p.add_argument(
        "--gene-list",
        default=None,
        help="Optional path to a text file with gene IDs (one per line). If set, only those genes are processed.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    limit = args.limit
    if limit == 0:
        limit = None

    run(
        out_path=out_path,
        limit=limit,
        batch_size=args.batch_size,
        timeout=args.timeout,
        sleep_base=args.sleep_base,
        sleep_jitter=args.sleep_jitter,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff_base,
        gene_list_path=args.gene_list,
    )

if __name__ == "__main__":
    main()

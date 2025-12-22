#!/usr/bin/env python3
"""
Download dictyBase Curator Notes (HTML + plain text) for a list of gene IDs.

Data source:
  BASE + "/gene/{gene_id}/gene/summary.json"

This script is intentionally simple (not over-engineered) and keeps the
same key variables/functions as in the notebook:
  - BASE
  - get_curator_notes_html
  - get_curator_notes_plain
  - GeneInput is fixed: Gene IDs are read from dictybase_files/DDB_G-curation_status.txt (http://dictybase.org/Downloads/)

Resumable output:
  - Writes to a Parquet file (default: curator_notes.parquet)
  - If the Parquet already exists, previously processed gene_ids are skipped.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import requests
from bs4 import BeautifulSoup


# Keep BASE unchanged (from the notebook)
BASE = "http://dictybase.org"


def make_session() -> requests.Session:
    """Create a requests session with a friendly User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": "dictybase-curator-notes/0.1"})
    return session


def get_curator_notes_html(
    gene_id: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> str | None:
    """
    Return curator notes as an HTML-ish string (with <i>, <br>, etc.),
    or None if 404 / no notes.

    NOTE: This reads the Curator Notes panel rendered as JSON.
    """
    url = f"{BASE}/gene/{gene_id}/gene/summary.json"
    r = session.get(url, timeout=timeout)

    if r.status_code == 404:
        return None
    r.raise_for_status()

    data = r.json()

    # The JSON is nested; Curator Notes live in the first column of the summary layout.
    try:
        col0 = data[0]["items"][0]                      # first column (Curator Notes)
        col_items = col0["content"][0]["items"]         # [title, body]
        content_row = col_items[1]                      # body after title "Curator Notes"
        tokens = content_row["content"][0]["items"]     # list of dict tokens: {"text": ...} or {"caption": ...}
    except (KeyError, IndexError, TypeError):
        return None

    fragments: List[str] = []
    for t in tokens:
        if isinstance(t, dict) and "text" in t:
            fragments.append(str(t["text"]))
        elif isinstance(t, dict) and "caption" in t:
            fragments.append(str(t["caption"]))

    html = "".join(fragments).strip()
    return html or None


def get_curator_notes_plain(
    gene_id: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> str | None:
    """
    Plain-text version of curator notes (HTML stripped).
    """
    html = get_curator_notes_html(gene_id, session=session, timeout=timeout)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def polite_sleep(base: float = 0.15, jitter: float = 0.10) -> None:
    """Small delay to avoid hammering the server."""
    time.sleep(base + random.random() * jitter)


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


def load_done_ids(out_path: Path) -> set[str]:
    """If output exists, load processed gene IDs to enable resume."""
    if not out_path.exists():
        return set()
    df_done = pl.read_parquet(out_path)
    if "gene_id" not in df_done.columns:
        return set()
    return set(df_done["gene_id"].to_list())


def append_parquet(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append a batch of rows to Parquet (or create the file if absent)."""
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return
    df_batch = pl.DataFrame(rows)

    if out_path.exists():
        df_batch.write_parquet(out_path, append=True)
    else:
        df_batch.write_parquet(out_path)


def run(
    out_path: Path,
    limit: int | None,
    batch_size: int,
    timeout: float,
    sleep_base: float,
    sleep_jitter: float,
) -> None:
    session = make_session()

    genes_status = load_genes_status()
    gene_series = genes_status.to_series()

    if limit is not None:
        gene_series = gene_series.head(limit)

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} genes already processed in {out_path}")
    else:
        print("Starting fresh")

    rows_buffer: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for gid in gene_series:
        gid = str(gid)
        if gid in done_ids:
            skipped += 1
            continue   # resume support

        html: Optional[str] = None
        plain: Optional[str] = None

        try:
            html = get_curator_notes_html(gid, session=session, timeout=timeout)
            if html:
                plain = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        except requests.RequestException as e:
            # Network errors, 5xx, etc. -> keep going
            print(f"{gid}: request failed ({e})")

        rows_buffer.append(
            {
                "gene_id": gid,
                "curator_notes_html": html,
                "curator_notes_plain": plain,
            }
        )
        processed += 1

        if len(rows_buffer) >= batch_size:
            append_parquet(out_path, rows_buffer)
            rows_buffer.clear()
            print(f"Saved {processed} new rows (skipped {skipped} already-done)")

        # Be polite to dictyBase (especially for large runs)
        polite_sleep(base=sleep_base, jitter=sleep_jitter)

    # flush remaining
    if rows_buffer:
        append_parquet(out_path, rows_buffer)
        print(f"Saved final batch. Total new rows: {processed} (skipped {skipped})")

    print(f"Done. Output: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape dictyBase Curator Notes into a Parquet file.")
    p.add_argument("--out", default="output/curator_notes.parquet", help="Output parquet path (default: curator_notes.parquet)")
    p.add_argument("--limit", type=int, default=10, help="How many genes to process (default: 10). Use 0 for all.")
    p.add_argument("--batch-size", type=int, default=200, help="Write to Parquet every N rows (default: 200).")
    p.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default: 15).")
    p.add_argument("--sleep-base", type=float, default=0.15, help="Base sleep between requests in seconds (default: 0.15).")
    p.add_argument("--sleep-jitter", type=float, default=0.10, help="Additional random jitter in seconds (default: 0.10).")
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
    )


if __name__ == "__main__":
    main()

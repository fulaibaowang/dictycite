#!/usr/bin/env python3
"""
Download dictyBase Curator Notes (HTML + plain text) for a list of gene IDs,
and additionally extract cited "claim sentences" with citation anchors.

Data source:
  BASE + "/gene/{gene_id}/gene/summary.json"

This script is intentionally simple (not over-engineered) and keeps the
same key variables/functions as in the notebook:
  - BASE
  - get_curator_notes_html
  - get_curator_notes_plain
  - GeneInput is fixed: Gene IDs are read from dictybase_files/DDB_G-curation_status.txt (http://dictybase.org/Downloads/)

Outputs:
  1) Gene-level notes (resumable):
     - gene_id, curator_notes_html, curator_notes_plain
     - default: output/dicty_gold_build/1_curator_notes.parquet

  2) Claim-level training rows (derived from curator notes):
     - gene_id, claim_plain, sentence_plain, sentence_markers, cited_sentence_marked,
       anchors, publication_ids, citation_captions, citation_years
     - default: output/dicty_gold_build/1_curator_claims.parquet

  3) Publication table (dedup by publication_id):
     - publication_id, caption_plain, year
     - default: output/dicty_gold_build/1_publications.parquet

Resume behavior:
  - Resume is keyed off the gene-level notes output: if a gene_id is already present
    in curator_notes.parquet, we skip processing that gene (and thus skip claims/pubs too).
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import requests
from bs4 import BeautifulSoup


# Keep BASE unchanged (from the notebook)
BASE = "http://dictybase.org"


# --- regex / helpers for claim extraction ---
YEAR_PAT = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
PUB_URL_PAT = re.compile(r"^/publication/(\d+)\b")
PUB_MARKER_PAT = re.compile(r"\[\[PUB:(\d+)\]\]")

# parenthetical OR marker; used for anchor-building
SENT_TOKEN_PAT = re.compile(r"\([^)]*\)|\[\[PUB:(\d+)\]\]")


def make_session() -> requests.Session:
    """Create a requests session with a friendly User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": "dictybase-curator-notes/0.1"})
    return session


def _strip_html_to_plain(s: str) -> str:
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)


def _dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _clean_text(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def _is_citation_only_sentence(sent: str) -> bool:
    """True if sentence contains PUB markers but otherwise has no real content."""
    if not PUB_MARKER_PAT.search(sent):
        return False
    s = PUB_MARKER_PAT.sub("", sent)
    s = re.sub(r"[\s\(\)\[\]\{\},;:.!?\-–—]+", "", s)
    return s == ""


def get_curator_notes_tokens(
    gene_id: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> List[Dict[str, Any]] | None:
    """
    Return Curator Notes token list from summary.json, or None on 404/missing.
    Tokens are dicts like {"text": "..."} or {"caption": "...", "url": "/publication/123", ...}
    """
    url = f"{BASE}/gene/{gene_id}/gene/summary.json"
    r = session.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()

    try:
        col0 = data[0]["items"][0]
        col_items = col0["content"][0]["items"]
        content_row = col_items[1]
        tokens = content_row["content"][0]["items"]
    except (KeyError, IndexError, TypeError):
        return None

    if not isinstance(tokens, list):
        return None
    return [t for t in tokens if isinstance(t, dict)]


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
    tokens = get_curator_notes_tokens(gene_id, session=session, timeout=timeout)
    if not tokens:
        return None

    fragments: List[str] = []
    for t in tokens:
        if "text" in t:
            fragments.append(str(t["text"]))
        elif "caption" in t:
            fragments.append(str(t["caption"]))

    html = "".join(fragments).strip()
    return html or None


def get_curator_notes_plain(
    gene_id: str,
    session: requests.Session,
    timeout: float = 15.0,
) -> str | None:
    """Plain-text version of curator notes (HTML stripped)."""
    html = get_curator_notes_html(gene_id, session=session, timeout=timeout)
    if html is None:
        return None
    return _strip_html_to_plain(html)


def _build_claim_and_anchors_from_marker_sentence(sent_markers: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    From a sentence containing [[PUB:####]] markers:
      - remove citation parentheticals (those containing markers)
      - remove bare markers
      - record anchor positions (char offsets) in the cleaned claim text

    Returns:
      claim_plain, anchors = [{"pos": int, "pub_ids":[...]}...]
    """
    raw_claim_parts: List[str] = []
    anchors_raw: Dict[int, List[str]] = {}  # pos_raw -> [pub_id strings]
    i = 0

    for m in SENT_TOKEN_PAT.finditer(sent_markers):
        start, end = m.start(), m.end()
        raw_claim_parts.append(sent_markers[i:start])

        chunk = sent_markers[start:end]

        # parenthetical block
        if chunk.startswith("(") and chunk.endswith(")"):
            pub_ids = PUB_MARKER_PAT.findall(chunk)
            if pub_ids:
                pos_raw = len("".join(raw_claim_parts))
                anchors_raw.setdefault(pos_raw, []).extend(pub_ids)
                # drop the entire citation parenthetical
            else:
                raw_claim_parts.append(chunk)  # keep other parentheses (e.g. gene lists)
        else:
            # bare marker [[PUB:####]]
            pid = m.group(1)
            if pid:
                pos_raw = len("".join(raw_claim_parts))
                anchors_raw.setdefault(pos_raw, []).append(pid)
            # drop the marker

        i = end

    raw_claim_parts.append(sent_markers[i:])
    raw_claim = "".join(raw_claim_parts)

    claim_plain = _clean_text(raw_claim)

    # remap raw anchor positions -> cleaned-text positions
    anchors: List[Dict[str, Any]] = []
    for pos_raw, pids in anchors_raw.items():
        prefix_clean = _clean_text(raw_claim[:pos_raw])
        pos_clean = len(prefix_clean)
        anchors.append(
            {
                "pos": pos_clean,
                "pub_ids": [int(x) for x in _dedup_keep_order(pids)],
            }
        )

    anchors = sorted(anchors, key=lambda d: d["pos"])
    return claim_plain, anchors


def extract_claims_and_pubs_from_tokens(
    gene_id: str,
    tokens: List[Dict[str, Any]] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build claim-level rows (with anchors) + publication rows from Curator Notes tokens.

    Returns:
      claims_rows, pub_rows
    """
    if not tokens:
        return [], []

    parts: List[str] = []
    pub_meta: Dict[str, Dict[str, Any]] = {}  # pub_id -> {caption_plain, year}

    for t in tokens:
        if "text" in t:
            parts.append(str(t["text"]))
            continue

        caption = t.get("caption")
        url = t.get("url")
        if caption is None:
            continue

        caption_str = str(caption)
        m = PUB_URL_PAT.match(str(url)) if url else None

        if m:
            pub_id = m.group(1)
            parts.append(f"[[PUB:{pub_id}]]")

            cap_plain = _strip_html_to_plain(caption_str)
            years = YEAR_PAT.findall(cap_plain)
            year = int(years[-1]) if years else None
            pub_meta[pub_id] = {"caption_plain": cap_plain, "year": year}
        else:
            # gene links and others: keep as text
            parts.append(caption_str)

    html_with_markers = "".join(parts)

    # normalize breaks for easier splitting
    tmp_plain = _strip_html_to_plain(
        html_with_markers.replace("<br>", ". ").replace("<br/>", ". ")
    )

    # naive sentence split
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", tmp_plain) if s.strip()]

    # merge citation-only sentences into previous
    merged: List[str] = []
    for s in raw:
        if merged and _is_citation_only_sentence(s):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)

    claims_rows: List[Dict[str, Any]] = []
    for sent_markers in merged:
        pub_ids = PUB_MARKER_PAT.findall(sent_markers)
        if not pub_ids:
            continue

        pub_ids_u = _dedup_keep_order(pub_ids)

        # claim + anchors (no citations)
        claim_plain, anchors = _build_claim_and_anchors_from_marker_sentence(sent_markers)

        # sentence_plain (citations rendered), plus marked sentence
        sent_plain = sent_markers
        cited_sentence_marked = sent_markers

        caps: List[str] = []
        years: List[int] = []
        for pid in pub_ids_u:
            meta = pub_meta.get(pid, {})
            cap_plain = meta.get("caption_plain", f"publication/{pid}")

            sent_plain = sent_plain.replace(f"[[PUB:{pid}]]", cap_plain)
            cited_sentence_marked = cited_sentence_marked.replace(f"[[PUB:{pid}]]", f"[CITE:{pid}]")

            caps.append(cap_plain)
            if meta.get("year") is not None:
                years.append(int(meta["year"]))

        claims_rows.append(
            {
                "gene_id": gene_id,
                "sentence_markers": sent_markers,
                "sentence_plain": sent_plain,
                "cited_sentence_marked": cited_sentence_marked,
                "claim_plain": claim_plain,
                "anchors": anchors,  # list[{"pos": int, "pub_ids": list[int]}]
                "publication_ids": [int(x) for x in pub_ids_u],
                "citation_captions": _dedup_keep_order(caps),
                "citation_years": _dedup_keep_order(years),
            }
        )

    pub_rows = [
        {
            "publication_id": int(pid),
            "caption_plain": meta.get("caption_plain"),
            "year": meta.get("year"),
        }
        for pid, meta in pub_meta.items()
    ]

    return claims_rows, pub_rows


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

    df_batch = pl.DataFrame(rows)

    if out_path.exists():
        try:
            df_existing = pl.read_parquet(out_path)
            df_combined = pl.concat([df_existing, df_batch], how="vertical")
            if unique_subset:
                # keep first occurrence
                df_combined = df_combined.unique(subset=unique_subset, keep="first")
            df_combined.write_parquet(out_path)
        except Exception as e:
            print(f"Warning: failed to append to {out_path} ({e}), writing batch only")
            df_batch.write_parquet(out_path)
    else:
        df_batch.write_parquet(out_path)


def run(
    out_path: Path,
    claims_out_path: Path,
    pubs_out_path: Path,
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

    notes_buffer: List[Dict[str, Any]] = []
    claims_buffer: List[Dict[str, Any]] = []
    pubs_buffer: List[Dict[str, Any]] = []

    processed = 0
    skipped = 0

    for gid in gene_series:
        gid = str(gid)
        if gid in done_ids:
            skipped += 1
            continue

        html: Optional[str] = None
        plain: Optional[str] = None
        tokens: Optional[List[Dict[str, Any]]] = None

        try:
            tokens = get_curator_notes_tokens(gid, session=session, timeout=timeout)
            if tokens:
                # build html/plain from tokens (same as before)
                fragments = []
                for t in tokens:
                    if "text" in t:
                        fragments.append(str(t["text"]))
                    elif "caption" in t:
                        fragments.append(str(t["caption"]))
                html = "".join(fragments).strip() or None
                if html:
                    plain = _strip_html_to_plain(html)

                # new: claims + pubs
                c_rows, p_rows = extract_claims_and_pubs_from_tokens(gid, tokens)
                claims_buffer.extend(c_rows)
                pubs_buffer.extend(p_rows)

        except requests.RequestException as e:
            print(f"{gid}: request failed ({e})")

        notes_buffer.append(
            {
                "gene_id": gid,
                "curator_notes_html": html,
                "curator_notes_plain": plain,
            }
        )
        processed += 1

        if len(notes_buffer) >= batch_size:
            append_parquet(out_path, notes_buffer, unique_subset=["gene_id"])
            notes_buffer.clear()

            # claims: dedup conservatively
            append_parquet(
                claims_out_path,
                claims_buffer,
                unique_subset=["gene_id", "sentence_markers"],
            )
            claims_buffer.clear()

            # pubs: dedup by publication_id
            append_parquet(pubs_out_path, pubs_buffer, unique_subset=["publication_id"])
            pubs_buffer.clear()

            print(f"Saved {processed} new genes (skipped {skipped} already-done)")

        polite_sleep(base=sleep_base, jitter=sleep_jitter)

    # flush remaining
    if notes_buffer:
        append_parquet(out_path, notes_buffer, unique_subset=["gene_id"])
        append_parquet(
            claims_out_path,
            claims_buffer,
            unique_subset=["gene_id", "sentence_markers"],
        )
        append_parquet(pubs_out_path, pubs_buffer, unique_subset=["publication_id"])
        print(f"Saved final batch. Total new genes: {processed} (skipped {skipped})")

    print("Done.")
    print(f"Notes output:  {out_path}")
    print(f"Claims output: {claims_out_path}")
    print(f"Pubs output:   {pubs_out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape dictyBase Curator Notes + claim anchors into Parquet files.")
    p.add_argument(
        "--out",
        default="output/dicty_gold_build/1_curator_notes.parquet",
        help="Output parquet path for gene-level notes (default: output/dicty_gold_build/1_curator_notes.parquet)",
    )
    p.add_argument(
        "--claims-out",
        default="output/dicty_gold_build/1_curator_claims.parquet",
        help="Output parquet path for claim-level rows (default: output/dicty_gold_build/1_curator_claims.parquet)",
    )
    p.add_argument(
        "--pubs-out",
        default="output/dicty_gold_build/1_publications.parquet",
        help="Output parquet path for publication table (default: output/dicty_gold_build/1_publications.parquet)",
    )
    p.add_argument("--limit", type=int, default=10, help="How many genes to process (default: 10). Use 0 for all.")
    p.add_argument("--batch-size", type=int, default=200, help="Write every N genes (default: 200).")
    p.add_argument("--timeout", type=float, default=15.0, help="Request timeout in seconds (default: 15).")
    p.add_argument("--sleep-base", type=float, default=0.15, help="Base sleep between requests in seconds (default: 0.15).")
    p.add_argument("--sleep-jitter", type=float, default=0.10, help="Additional random jitter in seconds (default: 0.10).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    claims_out_path = Path(args.claims_out)
    pubs_out_path = Path(args.pubs_out)

    limit = args.limit
    if limit == 0:
        limit = None

    run(
        out_path=out_path,
        claims_out_path=claims_out_path,
        pubs_out_path=pubs_out_path,
        limit=limit,
        batch_size=args.batch_size,
        timeout=args.timeout,
        sleep_base=args.sleep_base,
        sleep_jitter=args.sleep_jitter,
    )


if __name__ == "__main__":
    main()

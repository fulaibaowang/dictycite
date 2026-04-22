#!/usr/bin/env python3
"""
Build a gold-linked curator-notes **build dataset** (JSONL + stats).

For each gene_id that appears in the current gold standard (via 4a claim groups),
fetches dictyBase `summary.json`, stores a raw snapshot, builds a canonical ordered
`curator_notes_blocks` list (lossless mixed content), then derives marked/plain strings
and citation anchors. See docs/METHODS.md (Gold-linked notes build dataset).

Outputs (under output/dicty_gold_build/ by default):
  - 8_raw_notes_snapshot.jsonl
  - 8a_gold_linked_notes_build.jsonl
  - 8b_gold_linked_notes_examples.jsonl
  - 8c_build_provenance_stats.tsv
  - 8d_build_provenance_report.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import requests
from bs4 import BeautifulSoup

BASE = "http://dictybase.org"
PUB_URL_PAT = re.compile(r"^/publication/(\d+)\b")
GENE_URL_PAT = re.compile(r"^/gene/(DDB_G\d+)\b")
BR_PAT = re.compile(r"(<br\s*/?>)", re.IGNORECASE)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _strip_html_to_plain(s: str) -> str:
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)


def _clean_text(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def _cleanup_dangling_citation_punct(s: str) -> str:
    """
    Remove punctuation artifacts created when inline citation blocks are dropped
    from mixed text, e.g. '(.' / '(,' / '(;' / '()' and variants.
    Keep conservative and focused to avoid over-normalizing scientific text.
    """
    s = s.replace("\n", " \n ")
    # literal artifacts most often seen in output after citation drop
    s = s.replace("(.", ".")
    s = s.replace("(,", ",")
    s = s.replace("(;", ";")
    s = s.replace("(:", ":")
    s = s.replace("(!", "!")
    s = s.replace("(?", "?")
    # exact dangling open-paren before punctuation
    s = re.sub(r"\(\s*([,.;:!?])", r"\1", s)
    # empty parentheses that can be left after citation removal
    s = re.sub(r"\(\s*\)", "", s)
    # punctuation followed by a stray closing paren
    s = re.sub(r"([,.;:!?])\s*\)", r"\1", s)
    # collapse spaces introduced by replacements
    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace(" \n ", "\n")
    return s.strip()


def fetch_summary_json(
    gene_id: str,
    session: requests.Session,
    timeout: float,
) -> Tuple[int, Optional[Any], str]:
    url = f"{BASE}/gene/{gene_id}/gene/summary.json"
    r = session.get(url, timeout=timeout)
    body_text = r.text
    if r.status_code == 404:
        return 404, None, body_text
    r.raise_for_status()
    return r.status_code, r.json(), body_text


def extract_curator_tokens(data: Any) -> Optional[List[Dict[str, Any]]]:
    """Same path as dicty_curator_notes.get_curator_notes_tokens."""
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


def tokens_to_blocks(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Canonical lossless block stream: text (raw HTML fragments), citation, gene_link, break.
    """
    blocks: List[Dict[str, Any]] = []
    for t in tokens:
        if "text" in t:
            raw = str(t["text"])
            parts = BR_PAT.split(raw)
            for p in parts:
                if not p:
                    continue
                if BR_PAT.fullmatch(p):
                    blocks.append({"type": "break"})
                else:
                    blocks.append({"type": "text", "html": p})
            continue
        caption = t.get("caption")
        if caption is None:
            continue
        cap = str(caption)
        url = str(t.get("url") or "")
        mp = PUB_URL_PAT.match(url)
        if mp:
            blocks.append(
                {
                    "type": "citation",
                    "caption_html": cap,
                    "url": url,
                    "publication_id": int(mp.group(1)),
                }
            )
            continue
        mg = GENE_URL_PAT.match(url)
        if mg:
            blocks.append(
                {
                    "type": "gene_link",
                    "caption_html": cap,
                    "url": url,
                    "linked_gene_id": mg.group(1),
                }
            )
        else:
            blocks.append({"type": "gene_link", "caption_html": cap, "url": url})
    return blocks


def render_from_blocks(
    blocks: List[Dict[str, Any]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Derive curator_notes_marked_text, curator_notes_plain_text, citation_anchors.
    Anchors use UTF-16 code-unit offsets (Python slice indices = Unicode codepoints;
    for JSON consumers expecting JS string indices, document in METHODS — here we use
    Python str character indices as stable within this codebase).
    """
    marked_parts: List[str] = []
    plain_parts: List[str] = []
    anchors: List[Dict[str, Any]] = []
    mi, pi = 0, 0

    for bi, b in enumerate(blocks):
        typ = b.get("type")
        if typ == "text":
            frag_m = _strip_html_to_plain(b["html"])
            marked_parts.append(frag_m)
            plain_parts.append(frag_m)
            mi += len(frag_m)
            pi += len(frag_m)
        elif typ == "citation":
            pid = int(b["publication_id"])
            marker = f"[[PUB:{pid}]]"
            cap = _strip_html_to_plain(b["caption_html"])
            ms, me = mi, mi + len(marker)
            ps, pe = pi, pi + len(cap)
            anchors.append(
                {
                    "block_index": bi,
                    "publication_id": pid,
                    "stream": "marked",
                    "start": ms,
                    "end": me,
                    "placeholder": marker,
                }
            )
            anchors.append(
                {
                    "block_index": bi,
                    "publication_id": pid,
                    "stream": "plain",
                    "start": ps,
                    "end": pe,
                    "rendered_caption": cap,
                }
            )
            marked_parts.append(marker)
            plain_parts.append(cap)
            mi = me
            pi = pe
        elif typ == "gene_link":
            cap = _strip_html_to_plain(b["caption_html"])
            gid = b.get("linked_gene_id")
            if gid:
                gmark = f"[[GENE:{gid}]]"
                marked_parts.append(gmark)
                mi += len(gmark)
            else:
                marked_parts.append(cap)
                mi += len(cap)
            plain_parts.append(cap)
            pi += len(cap)
        elif typ == "break":
            marked_parts.append("\n")
            plain_parts.append("\n")
            mi += 1
            pi += 1

    marked = "".join(marked_parts)
    plain = _clean_text("".join(plain_parts))
    return marked, plain, anchors


def render_plain_no_citations(blocks: List[Dict[str, Any]]) -> str:
    """
    Plain text for generation examples with inline citation captions removed.
    Keeps text/gene mentions and line breaks; drops citation blocks entirely.
    """
    parts: List[str] = []
    for b in blocks:
        typ = b.get("type")
        if typ == "text":
            parts.append(_strip_html_to_plain(b.get("html", "")))
        elif typ == "gene_link":
            parts.append(_strip_html_to_plain(b.get("caption_html", "")))
        elif typ == "break":
            parts.append("\n")
        elif typ == "citation":
            # intentionally dropped
            continue
    return _cleanup_dangling_citation_punct(_clean_text("".join(parts)))


def _publication_ids_from_blocks(blocks: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for b in blocks:
        if b.get("type") == "citation":
            out.append(int(b["publication_id"]))
    return sorted(set(out))


def polite_sleep(base: float, jitter: float) -> None:
    time.sleep(base + random.random() * jitter)


def load_gold_questions(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[int]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    gids: List[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = str(q.get("query_id", q.get("id", "")))
            by_id[qid] = q
            try:
                gids.append(int(qid))
            except ValueError:
                continue
    return by_id, gids


def evidence_level_from_gold_question(q: Dict[str, Any]) -> str:
    """Aggregate 7a `docs[].evidence_level` for one gold question (group_claim_id)."""
    levels: List[str] = []
    for doc in q.get("docs") or []:
        if isinstance(doc, dict):
            el = doc.get("evidence_level")
            if el is not None and str(el).strip():
                levels.append(str(el).strip())
    unique = sorted(set(levels))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return " | ".join(unique)


def evidence_level_tokens(evidence_level: Optional[str]) -> set[str]:
    """Split 8b `claims[].evidence_level` (joined with ` | `) into token set."""
    lev = (evidence_level or "").strip()
    return {p.strip() for p in lev.split("|") if p.strip()}


_ABSTRACT_ONLY_LEVELS = frozenset({"abstract_supports_detail", "abstract_supports_core"})


def gold_linked_gene_ids(
    path_4a: Path,
    gold_group_ids: set[int],
) -> Tuple[set[str], Dict[str, set[int]]]:
    """Return unique gene ids and mapping gene_id -> set of group_claim_id."""
    df = pl.read_parquet(path_4a)
    df = df.filter(pl.col("group_claim_id").is_in(list(gold_group_ids)))
    gene_to_groups: Dict[str, set[int]] = defaultdict(set)
    all_genes: set[str] = set()
    for row in df.iter_rows(named=True):
        g = row["group_claim_id"]
        for part in str(row["gene_id"]).split(","):
            gid = part.strip()
            if gid:
                all_genes.add(gid)
                gene_to_groups[gid].add(int(g))
    return all_genes, dict(gene_to_groups)


def linked_claim_ids_by_gene(
    path_4a: Path,
    gold_group_ids: set[int],
) -> Dict[str, int]:
    """
    From 4a, count unique claim_id linked to gold groups for each split gene_id.
    gene_id can be comma-joined in 4a; we split and attribute the claim to each gene.
    """
    df = pl.read_parquet(path_4a).filter(pl.col("group_claim_id").is_in(list(gold_group_ids)))
    linked: Dict[str, set[int]] = defaultdict(set)
    for row in df.select(["gene_id", "claim_id"]).iter_rows(named=True):
        cid = int(row["claim_id"])
        for part in str(row["gene_id"]).split(","):
            gid = part.strip()
            if gid:
                linked[gid].add(cid)
    return {g: len(cids) for g, cids in linked.items()}


def load_gene_metadata(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        has_header=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build gold-linked curator notes dataset (JSONL).")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: infer from script location).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/output/dicty_gold_build).",
    )
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--sleep-base", type=float, default=0.2)
    p.add_argument("--sleep-jitter", type=float, default=0.1)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N genes (for testing).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Preload summary_json_raw from 8_raw_notes_snapshot.jsonl (last line per gene wins).",
    )
    p.add_argument(
        "--resume-build",
        action="store_true",
        help="Skip genes that already appear in 8a_gold_linked_notes_build.jsonl (append-only continuation).",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Append to existing JSONL/TSV instead of truncating first.",
    )
    p.set_defaults(overwrite=True)
    args = p.parse_args()

    root = args.repo_root.resolve() if args.repo_root else _repo_root_from_here()
    out_dir = (args.out_dir or (root / "output" / "dicty_gold_build")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    path_gold = out_dir / "7a_dicty_gold_llm_public.jsonl"
    path_4a = out_dir / "4a_claim_groups.parquet"
    path_claims = out_dir / "1_curator_claims.parquet"
    path_gene_info = root / "dictybase_files" / "gene_information.txt"
    path_pub_pmid = out_dir / "2_publication_id_pmid.csv"

    raw_snap = out_dir / "8_raw_notes_snapshot.jsonl"
    build_out = out_dir / "8a_gold_linked_notes_build.jsonl"
    build_b_out = out_dir / "8b_gold_linked_notes_examples.jsonl"
    stats_tsv = out_dir / "8c_build_provenance_stats.tsv"
    report_md = out_dir / "8d_build_provenance_report.md"

    if args.overwrite and not args.resume_build:
        for fp in (raw_snap, build_out, build_b_out, stats_tsv):
            if fp.exists():
                fp.unlink()
    elif args.overwrite and args.resume_build:
        for fp in (stats_tsv, build_b_out):
            if fp.exists():
                fp.unlink()

    already_built: set[str] = set()
    if args.resume_build and build_out.exists():
        with build_out.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gid = rec.get("gene_id")
                if gid:
                    already_built.add(str(gid))

    gold_by_id, _gold_ids_list = load_gold_questions(path_gold)
    gold_group_ids = {int(k) for k in gold_by_id.keys() if k.isdigit()}
    n_gold_groups = len(gold_group_ids)
    n_gold_doc_pairs = sum(
        len(gold_by_id[str(gid)].get("docs") or [])
        for gid in gold_group_ids
        if str(gid) in gold_by_id
    )

    genes_all, gene_to_groups = gold_linked_gene_ids(path_4a, gold_group_ids)
    linked_claim_counts = linked_claim_ids_by_gene(path_4a, gold_group_ids)
    gene_list = sorted(genes_all)
    if args.resume_build and already_built:
        gene_list = [g for g in gene_list if g not in already_built]
    if args.limit is not None:
        gene_list = gene_list[: args.limit]

    claims_df = pl.read_parquet(path_claims) if path_claims.exists() else pl.DataFrame()
    pub_pmid = (
        pl.read_csv(path_pub_pmid)
        if path_pub_pmid.exists()
        else pl.DataFrame(schema={"publication_id": pl.Int64, "pmid": pl.Utf8})
    )
    pmid_by_pub: Dict[int, str] = {}
    if pub_pmid.height > 0:
        for r in pub_pmid.iter_rows(named=True):
            pid = int(r["publication_id"])
            pm = r.get("pmid")
            if pm is not None and str(pm) not in ("", "NA", "nan"):
                pmid_by_pub[pid] = str(pm)

    gene_meta = load_gene_metadata(path_gene_info)
    # Columns: GENE ID, Gene Name, Synonyms, Gene products
    meta_cols = gene_meta.columns
    id_col, name_col = meta_cols[0], meta_cols[1]
    syn_col = meta_cols[2] if len(meta_cols) > 2 else None
    prod_col = meta_cols[3] if len(meta_cols) > 3 else None

    cached_raw: Dict[str, Any] = {}
    if (args.resume or args.resume_build) and raw_snap.exists():
        with raw_snap.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gid = rec.get("gene_id")
                if gid and rec.get("summary_json_raw") is not None:
                    cached_raw[str(gid)] = rec["summary_json_raw"]  # last line wins

    session = requests.Session()
    session.headers.update({"User-Agent": "dictycite-gold-linked-notes-build/0.1"})

    fetch_ok = 0
    fetch_404 = 0
    fetch_err = 0

    for gene_id in gene_list:
        groups = sorted(gene_to_groups.get(gene_id, set()))
        gold_groups_detail: List[Dict[str, Any]] = []
        gold_pubs: set[int] = set()
        for gc in groups:
            qid = str(gc)
            q = gold_by_id.get(qid, {})
            docs = q.get("docs") or []
            bodies = []
            pub_ids = []
            pmids = []
            for d in docs:
                if isinstance(d, dict):
                    pid = d.get("publication_id")
                    if pid is not None:
                        pub_ids.append(int(pid))
                        gold_pubs.add(int(pid))
                    pm = d.get("pmid")
                    if pm is not None:
                        pmids.append(str(pm))
            gold_groups_detail.append(
                {
                    "group_claim_id": gc,
                    "gold_question_id": qid,
                    "body": q.get("body"),
                    "n_docs": len(docs),
                    "publication_ids": sorted(set(pub_ids)),
                    "pmids": sorted(set(pmids)),
                }
            )

        summary_raw: Any = None
        status = None
        err_msg = None
        tokens: Optional[List[Dict[str, Any]]] = None

        if gene_id in cached_raw:
            summary_raw = cached_raw[gene_id]
            status = 200
            tokens = extract_curator_tokens(summary_raw)
        else:
            try:
                status, summary_raw, _txt = fetch_summary_json(
                    gene_id, session=session, timeout=args.timeout
                )
                if status == 404:
                    fetch_404 += 1
                    summary_raw = None
                    tokens = None
                else:
                    fetch_ok += 1
                    tokens = extract_curator_tokens(summary_raw) if summary_raw else None
            except requests.RequestException as e:
                fetch_err += 1
                err_msg = str(e)
                summary_raw = None
                tokens = None

        wrote_snap = False
        if gene_id not in cached_raw:
            snap_line = {
                "gene_id": gene_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "http_status": status,
                "error": err_msg,
                "summary_json_raw": summary_raw,
            }
            with raw_snap.open("a", encoding="utf-8") as rf:
                rf.write(json.dumps(snap_line, ensure_ascii=False) + "\n")
            wrote_snap = True

        blocks: List[Dict[str, Any]] = tokens_to_blocks(tokens) if tokens else []
        marked, plain, anchors = render_from_blocks(blocks) if blocks else ("", "", [])

        note_pubs = set(_publication_ids_from_blocks(blocks))
        pub_in_gold_and_note = sorted(note_pubs & gold_pubs)
        pub_in_note_only = sorted(note_pubs - gold_pubs)

        n_claim_rows = 0
        if claims_df.height > 0 and "gene_id" in claims_df.columns:
            n_claim_rows = claims_df.filter(pl.col("gene_id") == gene_id).height

        meta_row = gene_meta.filter(pl.col(id_col) == gene_id)
        gene_name = None
        synonyms = None
        gene_products = None
        if meta_row.height > 0:
            r0 = meta_row.row(0, named=True)
            gene_name = r0.get(name_col)
            if syn_col:
                synonyms = r0.get(syn_col)
            if prod_col:
                gene_products = r0.get(prod_col)

        n_gold_groups_for_gene = len(groups)
        n_linked_claim_ids = int(linked_claim_counts.get(gene_id, 0))
        n_non_gold_claim_rows = max(0, int(n_claim_rows) - n_linked_claim_ids)
        pmids_for_note_pubs = [pmid_by_pub[p] for p in note_pubs if p in pmid_by_pub]

        build_row = {
            "gene_id": gene_id,
            "gene_name": gene_name,
            "synonyms": synonyms,
            "gene_products": gene_products,
            "summary_json_raw": summary_raw,
            "curator_notes_blocks": blocks,
            "curator_notes_marked_text": marked,
            "curator_notes_plain_text": plain,
            "citation_anchors": anchors,
            "gold_linkage": {
                "group_claim_ids": groups,
                "gold_groups": gold_groups_detail,
            },
            "coverage": {
                "n_gold_groups_for_gene": n_gold_groups_for_gene,
                "n_distinct_publication_ids_in_notes": len(note_pubs),
                "n_distinct_publication_ids_in_gold_docs": len(gold_pubs),
                "n_publication_ids_in_both_notes_and_gold": len(pub_in_gold_and_note),
                "publication_ids_in_notes_only": pub_in_note_only,
                "n_curator_claim_rows_in_1_curator_claims": n_claim_rows,
                "n_linked_claim_ids_from_4a": n_linked_claim_ids,
                "n_non_gold_claim_rows_estimate": n_non_gold_claim_rows,
                "n_note_publication_ids_with_pmid_mapping": len(pmids_for_note_pubs),
            },
        }
        with build_out.open("a", encoding="utf-8") as bf:
            bf.write(json.dumps(build_row, ensure_ascii=False) + "\n")

        if gene_id not in cached_raw and wrote_snap:
            polite_sleep(args.sleep_base, args.sleep_jitter)

    stats_rows: List[Dict[str, Any]] = []
    if build_out.exists():
        with build_out.open(encoding="utf-8") as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                blocks_r = rec.get("curator_notes_blocks") or []
                cov = rec.get("coverage") or {}
                gl = rec.get("gold_linkage") or {}
                gene_id = rec.get("gene_id")
                n_curator_claim_rows = int(cov.get("n_curator_claim_rows_in_1_curator_claims") or 0)
                n_linked_claim_ids = int(
                    cov.get("n_linked_claim_ids_from_4a") or linked_claim_counts.get(gene_id, 0)
                )
                n_non_gold_claim_rows = max(0, n_curator_claim_rows - n_linked_claim_ids)
                stats_rows.append(
                    {
                        "gene_id": gene_id,
                        "n_gold_groups": cov.get("n_gold_groups_for_gene"),
                        "n_blocks": len(blocks_r),
                        "n_citation_blocks": sum(
                            1 for b in blocks_r if b.get("type") == "citation"
                        ),
                        "n_curator_claim_rows": n_curator_claim_rows,
                        "n_linked_claim_ids": n_linked_claim_ids,
                        "n_non_gold_claim_rows": n_non_gold_claim_rows,
                        "non_gold_claim_fraction": (
                            (n_non_gold_claim_rows / n_curator_claim_rows)
                            if n_curator_claim_rows > 0
                            else 0.0
                        ),
                        "n_note_publication_ids": cov.get(
                            "n_distinct_publication_ids_in_notes"
                        ),
                        "n_gold_publication_ids": cov.get(
                            "n_distinct_publication_ids_in_gold_docs"
                        ),
                        "n_pub_in_notes_and_gold": cov.get(
                            "n_publication_ids_in_both_notes_and_gold"
                        ),
                        "plain_text_len": len(rec.get("curator_notes_plain_text") or ""),
                        "n_gold_group_ids_listed": len(gl.get("group_claim_ids") or []),
                    }
                )
    if stats_rows:
        pl.DataFrame(stats_rows).write_csv(stats_tsv, separator="\t")

    # Build 8b: gene-level examples with local claim IDs.
    # Inclusion rule (requested): genes with >2 claims AND zero non-gold claims.
    n_8b_rows = 0
    n_8b_claims_total = 0
    n_8b_claims_exact_detail = 0
    n_8b_claims_abstract_only = 0
    n_8b_claims_with_fulltext = 0
    n_8b_genes_all_exact_detail = 0
    n_8b_genes_abstract_only = 0
    n_8b_genes_with_fulltext = 0
    if build_out.exists() and stats_tsv.exists():
        stats_df = pl.read_csv(stats_tsv, separator="\t").select(
            ["gene_id", "n_curator_claim_rows", "n_non_gold_claim_rows"]
        )
        keep_genes = set(
            stats_df.filter(
                (pl.col("n_curator_claim_rows") > 2)
                & (pl.col("n_non_gold_claim_rows") == 0)
            )["gene_id"].to_list()
        )

        with build_out.open(encoding="utf-8") as sf, build_b_out.open("a", encoding="utf-8") as bf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gene_id = rec.get("gene_id")
                if gene_id not in keep_genes:
                    continue

                gold_groups = (rec.get("gold_linkage") or {}).get("gold_groups") or []
                if not gold_groups:
                    continue

                # Stable local claim IDs c1..cn by group_claim_id ordering.
                gold_groups_sorted = sorted(
                    gold_groups,
                    key=lambda g: (int(g.get("group_claim_id") or 0), str(g.get("gold_question_id") or "")),
                )
                claims: List[Dict[str, Any]] = []
                gold_support: List[Dict[str, Any]] = []

                # Aggregate documents at gene level: unique publication_id with pmid.
                docs_by_pub: Dict[int, Dict[str, Any]] = {}

                for i, g in enumerate(gold_groups_sorted, start=1):
                    claim_id = f"c{i}"
                    body = (g.get("body") or "").strip()
                    q = gold_by_id.get(str(g.get("group_claim_id")), {})
                    claims.append(
                        {
                            "claim_id": claim_id,
                            "text": body,
                            "importance": "",
                            "evidence_level": evidence_level_from_gold_question(q),
                        }
                    )

                    pmids = sorted(
                        {
                            str(p).strip()
                            for p in (g.get("pmids") or [])
                            if str(p).strip() not in ("", "NA", "null", "None")
                        }
                    )
                    gold_support.append(
                        {
                            "claim_id": claim_id,
                            "pmids": pmids,
                        }
                    )

                    pub_ids = g.get("publication_ids") or []
                    q_docs = q.get("docs") or []
                    q_docs_by_pub: Dict[int, Dict[str, Any]] = {}
                    for doc in q_docs:
                        if not isinstance(doc, dict):
                            continue
                        pid_val = doc.get("publication_id")
                        if pid_val is None:
                            continue
                        try:
                            q_docs_by_pub[int(pid_val)] = doc
                        except (TypeError, ValueError):
                            continue
                    for j, pid in enumerate(pub_ids):
                        try:
                            pub_id = int(pid)
                        except (TypeError, ValueError):
                            continue
                        pmid = ""
                        if j < len(pmids):
                            pmid = pmids[j]
                        doc_full = q_docs_by_pub.get(pub_id, {})
                        title = str(doc_full.get("title") or "")
                        abstract_text = str(doc_full.get("abstract_clean") or "")
                        docs_by_pub[pub_id] = {
                            "publication_id": pub_id,
                            "pmid": pmid,
                            "title": title,
                            "text": abstract_text,
                        }

                blocks_for_gene = rec.get("curator_notes_blocks") or []
                example = {
                    "gene_id": gene_id,
                    "gene_name": rec.get("gene_name"),
                    "curator_notes": render_plain_no_citations(blocks_for_gene),
                    "claims": claims,
                    "documents": [docs_by_pub[k] for k in sorted(docs_by_pub)],
                    "gold_support": gold_support,
                    "provenance": {
                        "raw_snapshot_ref": f"output/dicty_gold_build/8_raw_notes_snapshot.jsonl#gene_id={gene_id}",
                        "group_claim_ids": sorted(
                            [
                                int(g.get("group_claim_id"))
                                for g in gold_groups_sorted
                                if g.get("group_claim_id") is not None
                            ]
                        ),
                    },
                }
                bf.write(json.dumps(example, ensure_ascii=False) + "\n")
                n_8b_rows += 1

                n_8b_claims_total += len(claims)
                for c in claims:
                    el = (c.get("evidence_level") or "").strip()
                    toks = evidence_level_tokens(c.get("evidence_level"))
                    if el == "abstract_supports_detail":
                        n_8b_claims_exact_detail += 1
                    if toks and toks <= _ABSTRACT_ONLY_LEVELS:
                        n_8b_claims_abstract_only += 1
                    if "needs_fulltext" in toks:
                        n_8b_claims_with_fulltext += 1

                if claims:
                    per_toks = [evidence_level_tokens(c.get("evidence_level")) for c in claims]
                    if all(
                        (c.get("evidence_level") or "").strip() == "abstract_supports_detail"
                        for c in claims
                    ):
                        n_8b_genes_all_exact_detail += 1
                    if all(t and t <= _ABSTRACT_ONLY_LEVELS for t in per_toks):
                        n_8b_genes_abstract_only += 1
                    if any("needs_fulltext" in t for t in per_toks):
                        n_8b_genes_with_fulltext += 1

    report_lines = [
        "# Gold-linked notes build — provenance report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Repository root: `{root}`",
        f"- Gold JSON: `{path_gold}`",
        f"- Linkage: `{path_4a}`",
        f"- Curator claims (stats only): `{path_claims}`",
        f"- Gene metadata: `{path_gene_info}`",
        "",
        "## Global counts",
        "",
        f"- Gold groups (questions): **{n_gold_groups}**",
        f"- Gold claim–document pairs (sum of docs): **{n_gold_doc_pairs}**",
        f"- Distinct genes linked to gold (from 4a): **{len(genes_all)}**",
        f"- Genes written in this run: **{len(gene_list)}**",
        f"- HTTP fetch OK: **{fetch_ok}**, 404: **{fetch_404}**, errors: **{fetch_err}**",
        "",
        "## Outputs",
        "",
        f"- `{raw_snap.relative_to(root)}` — one JSON object per line; includes `summary_json_raw`.",
        f"- `{build_out.relative_to(root)}` — build dataset rows (blocks + derived text + gold linkage).",
        f"- `{build_b_out.relative_to(root)}` — gene examples for genes with >2 claims and zero non-gold claims.",
        f"- `{stats_tsv.relative_to(root)}` — per-gene metrics.",
        "",
        "## Schema notes",
        "",
        "- **curator_notes_blocks**: ordered list; `citation` blocks include dictyBase `publication_id` for PMID joins.",
        "- **curator_notes_marked_text**: derived; citations as `[[PUB:<id>]]`, gene links as `[[GENE:<id>]]` when URL parses.",
        "- **curator_notes_plain_text**: derived; citations rendered as stripped author–year captions.",
        "- **citation_anchors**: list of spans (`stream` = `marked` | `plain`) with `start`/`end` (Python string indices).",
        "- **8b local IDs**: `claims[].claim_id` are local (`c1..cn`) within each gene example; not global IDs.",
        "- **8b evidence_level**: `claims[].evidence_level` copied from 7a `questions[].docs[].evidence_level`; if a question has multiple docs with different levels, unique values are sorted and joined with ` | `.",
        f"- **8b row count**: **{n_8b_rows}**.",
        "",
        "## 8b evidence-level summary",
        "",
        f"Per-gene categories partition the **{n_8b_rows}** genes that passed the 8b filter: each gene has at least one gold claim, and it falls in **abstract-only** iff every claim’s token set is non-empty and ⊆ `abstract_supports_detail` ∪ `abstract_supports_core`; otherwise it has **needs_fulltext** on at least one claim.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Genes in 8b (one curator-note example per gene) | {n_8b_rows} |",
        f"| Total claims (sum over genes) | {n_8b_claims_total} |",
        f"| Genes: every claim is exactly `abstract_supports_detail` | {n_8b_genes_all_exact_detail} |",
        f"| Genes: every claim uses only `abstract_supports_detail` and/or `abstract_supports_core` (no `needs_fulltext`) | {n_8b_genes_abstract_only} |",
        f"| Genes: at least one claim includes `needs_fulltext` | {n_8b_genes_with_fulltext} |",
        "",
        "| Claims (row-level) | Count |",
        "| --- | ---: |",
        f"| Claim string is exactly `abstract_supports_detail` | {n_8b_claims_exact_detail} |",
        f"| Claim tokens ⊆ `abstract_supports_detail` ∪ `abstract_supports_core` (non-empty) | {n_8b_claims_abstract_only} |",
        f"| Claim includes token `needs_fulltext` | {n_8b_claims_with_fulltext} |",
        "",
    ]
    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"Raw snapshot: {raw_snap}")
    print(f"Build JSONL:  {build_out}")
    print(f"Build 8b:     {build_b_out}")
    print(f"Stats TSV:    {stats_tsv}")
    print(f"Report:       {report_md}")


if __name__ == "__main__":
    main()

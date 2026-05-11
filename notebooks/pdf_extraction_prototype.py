# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dicty (Python 3.14 venv)
#     language: python
#     name: dicty-py314
# ---

# %% [markdown]
# # PDF extraction prototype
#
# Compare extractors on a small sample of dictybase PDFs and prototype the cleaning pipeline.
# Goal: pick one extractor + cleaning chain for the full ~228-PDF processing run.
#
# Sample (`abstract_insufficient` unless noted):
# - `10373524` (1999) — pdfs/
# - `16367873` (2006) — pdfs/
# - `24550398` (2014) — pdfs/
# - `26359303` (2015, abstract_supports_detail control) — pdfs/
# - `29626371` (2018, manual)
#
# Edge cases for cleaning stress-test (not in example files; included to exercise tiers 1-2):
# - `9564522` — *Benchmarks* multi-article PDF (last page is from a different paper)
# - `27663234` — Elsevier preproof: page 1 is metadata, "ACCEPTED MANUSCRIPT" watermark on every page

# %%
from pathlib import Path
import fitz  # pymupdf
import pdfplumber

PDFS_DIR = Path("/Users/yun/Documents/dictybase_papers/pdfs")
MANUAL_DIR = Path("/Users/yun/Documents/dictybase_papers/manual")
OUT_DIR = Path("/Users/yun/develop/dictycite/output/pdf_extraction/prototype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("10373524", PDFS_DIR, "1999 abstract_insufficient"),
    ("16367873", PDFS_DIR, "2006 abstract_insufficient"),
    ("24550398", PDFS_DIR, "2014 abstract_insufficient"),
    ("26359303", PDFS_DIR, "2015 abstract_supports_detail (control)"),
    ("29626371", MANUAL_DIR, "2018 abstract_insufficient (manual)"),
    ("9564522",  PDFS_DIR, "edge: Benchmarks multi-article"),
    ("27663234", PDFS_DIR, "edge: Elsevier accepted manuscript"),
]


# %% [markdown]
# ## Extractor 1 — PyMuPDF, naive `get_text()`
# Reads text in raw PDF stream order. Fast, but bad for two-column layouts.

# %%
def extract_pymupdf_naive(pdf_path: Path) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


# %% [markdown]
# ## Extractor 2 — PyMuPDF, blocks mode with column-aware sort
# Groups text into blocks, sorts top-to-bottom within each column band by detecting two-column splits via x-coordinate.

# %%
def extract_pymupdf_blocks(pdf_path: Path) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            if not text_blocks:
                continue
            page_w = page.rect.width
            mid = page_w / 2
            # split into left/right column buckets by block center
            left, right = [], []
            for b in text_blocks:
                cx = (b[0] + b[2]) / 2
                (left if cx < mid else right).append(b)
            # if right column is sparse it's probably single-column; merge
            if len(right) < len(left) * 0.3:
                ordered = sorted(text_blocks, key=lambda b: (b[1], b[0]))
            else:
                ordered = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
            parts.append("\n".join(b[4].strip() for b in ordered))
    return "\n\n".join(parts)


# %% [markdown]
# ## Extractor 3 — pdfplumber with `layout=True`
# Layout-aware text extraction; preserves visual ordering, generally handles columns well.

# %%
def extract_pdfplumber(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(layout=False, x_tolerance=2, y_tolerance=2) or ""
            parts.append(txt)
    return "\n".join(parts)


# %% [markdown]
# ## Run all extractors on the sample

# %%
EXTRACTORS = {
    "pymupdf_naive": extract_pymupdf_naive,
    "pymupdf_blocks": extract_pymupdf_blocks,
    "pdfplumber": extract_pdfplumber,
}

stats = []  # rows: pmid, extractor, n_chars, n_lines, t_sec
import time

for pmid, src_dir, label in SAMPLES:
    pdf_path = src_dir / f"{pmid}.pdf"
    print(f"\n=== {pmid} ({label}) — {pdf_path} ===")
    if not pdf_path.exists():
        print("  MISSING")
        continue
    for ex_name, ex_fn in EXTRACTORS.items():
        t0 = time.time()
        try:
            text = ex_fn(pdf_path)
        except Exception as e:
            print(f"  {ex_name}: ERROR {e}")
            continue
        dt = time.time() - t0
        out_path = OUT_DIR / f"{pmid}__{ex_name}.txt"
        out_path.write_text(text)
        n_chars = len(text)
        n_lines = text.count("\n")
        stats.append({"pmid": pmid, "extractor": ex_name, "n_chars": n_chars, "n_lines": n_lines, "t_sec": round(dt, 2)})
        print(f"  {ex_name:18s}: {n_chars:>7d} chars, {n_lines:>5d} lines, {dt:5.2f}s -> {out_path.name}")

# %% [markdown]
# ## Summary table

# %%
import pandas as pd
df = pd.DataFrame(stats)
df_pivot = df.pivot(index="pmid", columns="extractor", values="n_chars")
print("Char count per extractor per PDF:")
print(df_pivot)
print("\nTimings (sec):")
print(df.pivot(index="pmid", columns="extractor", values="t_sec"))

# %% [markdown]
# ## Quick quality probes
#
# Things to look for in each output file:
# 1. **Two-column ordering**: does paragraph N+1 follow paragraph N, or do columns get interleaved line-by-line?
# 2. **References section**: cleanly separable by a `References` / `Bibliography` heading?
# 3. **Figure captions**: kept (good — they're often citable claims) or merged into body awkwardly?
# 4. **Headers/footers**: page numbers, journal name on every page polluting body text?
# 5. **Hyphenation**: line-break hyphens stitched (`co-\noperate` → `cooperate`)?
#
# Open the `.txt` files in `output/pdf_extraction_prototype/` side-by-side to eyeball.

# %%
# Snippet preview — first 2000 chars from each extractor for the most-recent PDF
target = "29626371"
for ex_name in EXTRACTORS:
    p = OUT_DIR / f"{target}__{ex_name}.txt"
    if p.exists():
        print(f"\n--- {ex_name} ---")
        print(p.read_text()[:2000])
        print("...")

# %% [markdown]
# # Cleaning pipeline
#
# Verdict from the comparison above: **PyMuPDF naive** is the winner — fastest, handles two-column
# layouts correctly via the underlying PDF text stream order, and pdfplumber actively breaks on
# modern multi-column papers.
#
# Now layer in cleaning. Two tiers:
# - **Tier 1 (mechanical)**: ligatures, `(cid:N)`, de-hyphenation, repeated header/footer detection
# - **Tier 2 (structural)**: drop References / Acknowledgments; tag figure/table captions

# %% [markdown]
# ## Tier 1.a — Page-level extract that preserves page boundaries
#
# We need pages as a list (not concatenated) so we can detect repeated headers/footers across pages.

# %%
def extract_pages_pymupdf(pdf_path: Path) -> list[str]:
    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


# %% [markdown]
# ## Tier 1.b — Repeated header/footer stripping
#
# Look at the first and last few non-empty lines of each page; any line that appears on >=60% of
# pages is treated as a running header/footer and removed. Catches: journal title, "ACCEPTED
# MANUSCRIPT", page number bands.

# %%
import re
from collections import Counter

def strip_repeated_headers_footers(
    pages: list[str],
    edge_frac: float = 0.6,
    edge_lines: int = 3,
    watermark_frac: float = 0.8,
) -> list[str]:
    """Drop two classes of repeated lines:
    - **Edge repeats**: lines appearing at top or bottom of >=edge_frac of pages
      (running journal title, page-number bands).
    - **Watermarks**: any line appearing anywhere on >=watermark_frac of pages
      (e.g., 'ACCEPTED MANUSCRIPT' which can appear at varying y-positions).
    """
    if len(pages) < 3:
        return pages
    head_counter, foot_counter, all_counter = Counter(), Counter(), Counter()
    page_lines = []
    for txt in pages:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        page_lines.append(lines)
        head_counter.update(lines[:edge_lines])
        foot_counter.update(lines[-edge_lines:])
        all_counter.update(set(lines))  # dedup within-page so single line counts once per page
    n = len(pages)
    edge_threshold = max(2, int(n * edge_frac))
    watermark_threshold = max(2, int(n * watermark_frac))
    repeated_edge = {ln for ln, c in (head_counter + foot_counter).items()
                     if c >= edge_threshold and len(ln) > 2}
    repeated_watermark = {ln for ln, c in all_counter.items()
                          if c >= watermark_threshold and len(ln) > 2}
    repeated = repeated_edge | repeated_watermark
    page_num_re = re.compile(r"^\s*\d{1,4}\s*$")
    cleaned = []
    for lines in page_lines:
        kept = [ln for ln in lines if ln not in repeated and not page_num_re.match(ln)]
        cleaned.append("\n".join(kept))
    return cleaned


# %% [markdown]
# ## Tier 1.c — Character-level cleanups
# Ligatures, `(cid:N)`, de-hyphenation, whitespace collapse.

# %%
LIGATURE_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff",
    "ﬅ": "ft", "ﬆ": "st",
    "­": "",   # soft hyphen
}

def clean_chars(text: str) -> str:
    for k, v in LIGATURE_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"\(cid:\d+\)", "", text)
    return text


def dehyphenate_linebreaks(text: str) -> str:
    """Join 'co-\nword' → 'coword' when the prefix doesn't look like a real compound."""
    # Only join when next line starts lowercase and prefix isn't a known short prefix (e.g., 't-test')
    def repl(m):
        before, after = m.group(1), m.group(2)
        # Don't merge if before is very short (likely a real hyphen): "t-test", "X-ray"
        if len(before) <= 1:
            return f"{before}-\n{after}"
        return f"{before}{after}"
    return re.sub(r"(\w+)-\n(\w+)", repl, text)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Tier 1.d — leading boilerplate strip
# Lines matching any of these patterns are dropped from the start of the body until we hit
# a line that doesn't match (then we stop and keep everything from there).
# Conservative: only matches obvious metadata; never drops something that could be content.
_NOISE_PATTERNS = [
    r"^doi[:\s]\s*10\.\d",                                  # DOI line
    r"^\d{4}-\d{4}/\d+/?\$",                                # ISSN/$ pattern (e.g. 0270-7306/99/$04.0010)
    r"copyright\s+©|©\s*\d{4}|all\s+rights\s+reserved",     # copyright (©)
    r"^[©0]\s*\d{4}\s+by\s+the",                            # OCR-mangled "© 1993 by The..."
    r"^vol\.?\s*\d|^vol\s+\d",                              # Vol. N
    r"^pp\.\s*\d|^p\.\s*\d+\s*[-–]\s*\d+\s*$",              # pp. N-N
    r"^no\.\s*\d+\s*$",                                     # No. N
    r"^issue\s+of\s+",                                      # "Issue of ..."
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4},?\s+p\.?\s*\d",   # "July 1999, p. 4750-4756"
    r"first\s+published\s+online",
    r"^printed\s+in\s+",
    r"^\(?received\s+(for\s+publication|on|in|by)|^\(?received\s+\d",
    r"^submitted\s+\w+\s+\d+",
    r"^accepted\s+\w+\s+\d+|^accepted\s+for\s+",
    r"^revised\s+(form|\w+\s+\d+)",
    r"^published\s+(in\s+press|online|by)",
    r"^communicated\s+by",
    r"^edited\s+by",
    r"^\*?\s*for\s+correspondence",                          # "*For correspondence" line(s)
    r"^e[-]?\s?mail[:\s]",                                   # "E-mail:" or "E mail"
    r"^tel\.?[:\s]|^fax\.?[:\s]|^phone[:\s]",                # contact
    r"^proc\.?\s+natl\.?\s+acad",                            # PNAS banner
    r"^the\s+journal\s+of\s+biological\s+chemistry",         # JBC banner
    r"^journal\s+of\s+\w",                                   # generic "Journal of X" banner
    r"^molecular\s+and\s+cellular\s+biology",
    r"^cellular\s+microbiology",
    r"^current\s+biology\s+\d",                              # Current Biology banner with vol number
    r"^eukaryotic\s+cell",
    r"^developmental\s+biology\s*$|^developmental\s+cell\s*$",
    r"^report\s*$|^article\s*$|^original\s+article\s*$|^research\s+article\s*$",
    r"^brief\s+communication\s*$|^letter\s*$|^review\s*$",
    r"^research\s*$|^methods\s*$",
    r"^\s*\d{1,4}\s*$",                                     # bare page numbers
]
_NOISE_RE = re.compile("|".join(f"(?:{p})" for p in _NOISE_PATTERNS), re.IGNORECASE)
# All-caps banner detection — requires 3+ consecutive all-caps words.
# Single all-caps tokens like "CBP1", "PIP3", "WD40" must NOT trigger this.
_ALLCAPS_BANNER_RE = re.compile(
    r"^[A-Z][A-Z0-9\.,'&-]+(?:\s+[A-Z][A-Z0-9\.,'&-]+){2,}"
)

# Lines that are almost-certainly a single-line journal banner: short, mostly digits/punctuation
_JUNK_LINE_RE = re.compile(r"^[\d\s,\.:;\-–—()\[\]/]+$")


def strip_leading_boilerplate(text: str, max_lines_to_check: int = 30) -> str:
    """Drop journal/copyright/receipt boilerplate from the start of the body.
    Stops at the first line that doesn't match any noise pattern."""
    lines = text.split("\n")
    keep_from = 0
    for i, ln in enumerate(lines[:max_lines_to_check]):
        s = ln.strip()
        if not s:
            keep_from = i + 1
            continue
        if _NOISE_RE.search(s) or _JUNK_LINE_RE.match(s) or _ALLCAPS_BANNER_RE.match(s):
            keep_from = i + 1
            continue
        break
    return "\n".join(lines[keep_from:])


def tier1_clean(pages: list[str]) -> str:
    pages = strip_repeated_headers_footers(pages)
    text = "\n\n".join(pages)
    text = clean_chars(text)
    text = dehyphenate_linebreaks(text)
    text = collapse_whitespace(text)
    text = strip_leading_boilerplate(text)
    return text


# %% [markdown]
# ## Tier 2 — Structural cuts
#
# Locate canonical section headings and split off References/Acknowledgments. Tag figure/table captions.

# %%
# References heading: line on its own, case-insensitive, with optional numbering.
# Variants observed in the corpus: "References", "References and Notes" (Science),
# "References Cited", "Reference List" (some old papers), "Bibliography",
# "Literature Cited", "Works Cited".
REFS_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?("
    r"references?(?:\s+(?:and\s+notes|list|cited))?"
    r"|reference\s+list"
    r"|bibliography"
    r"|literature\s+cited"
    r"|works\s+cited"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ACK_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(acknowledg(e?)ments?|acknowledg(e?)ment)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table)\s*\d+[.:]\s",
    re.IGNORECASE | re.MULTILINE,
)


def _last_match(regex: re.Pattern, text: str) -> re.Match | None:
    """Return the last match of regex in text. Avoids cutting at TOC entries / table column
    headers that happen to be 'References' / 'Acknowledgments' on their own line."""
    last = None
    for m in regex.finditer(text):
        last = m
    return last


def _detect_unlabeled_refs_run(text: str, min_consecutive: int = 5) -> int | None:
    """Detect a numbered reference list with no preceding 'References' heading.

    Returns the offset where the numbered run starts, or None if not detected.
    Used as a fallback for papers (Nature/PNAS-style) that emit numbered refs without
    an explicit heading line.
    """
    # Lines like "12. Smith J, ..." or "12 Smith J, ..."
    num_line_re = re.compile(r"^\s*(\d{1,3})[.\s]\s*[A-Z]", re.MULTILINE)
    matches = list(num_line_re.finditer(text))
    if len(matches) < min_consecutive:
        return None
    # Walk from the end; find the longest run of consecutive numbers (1..N)
    # within proximity (each match within 800 chars of the next).
    best_start = None
    run = []
    last_offset = None
    last_n = None
    for m in matches:
        n = int(m.group(1))
        if last_offset is None or (m.start() - last_offset < 800 and n == (last_n or 0) + 1):
            run.append(m)
        else:
            if len(run) >= min_consecutive:
                best_start = run[0].start()
            run = [m]
        last_offset = m.start()
        last_n = n
    if len(run) >= min_consecutive:
        best_start = run[0].start()
    # Only accept if it sits in the back half of the document (refs are at the end)
    if best_start is not None and best_start > 0.5 * len(text):
        return best_start
    return None


def split_structural(text: str) -> dict:
    """Return {body, references, acknowledgments, captions: [...]}.
    Body excludes References + Acknowledgments. Captions are extracted as separate items
    (for chunk tagging later)."""
    refs_m = _last_match(REFS_RE, text)
    ack_m = _last_match(ACK_RE, text)
    cut_at = len(text)
    refs = ""
    acks = ""
    refs_via = None
    if refs_m:
        cut_at = min(cut_at, refs_m.start())
        refs = text[refs_m.end():].strip()
        refs_via = "heading"
    else:
        # Fallback: detect a numbered reference list at the document tail
        run_start = _detect_unlabeled_refs_run(text)
        if run_start is not None:
            cut_at = min(cut_at, run_start)
            refs = text[run_start:].strip()
            refs_via = "unlabeled_run"
    # Only honor Acknowledgments match if it's near the end (avoids matching e.g., a Chapter 6
    # heading "Acknowledgments" in a TOC).  Pick the last ack match that occurs before refs.
    if ack_m and ack_m.start() < cut_at:
        ack_end = refs_m.start() if refs_m else len(text)
        # Discard if ack heading is suspiciously far before refs (>50% of doc apart) — likely TOC
        if not refs_m or (ack_end - ack_m.start()) < 0.5 * len(text):
            acks = text[ack_m.end():ack_end].strip()
            cut_at = min(cut_at, ack_m.start())
    body = text[:cut_at].strip()

    captions = [m.group(0).strip() for m in CAPTION_RE.finditer(body)]
    return {"body": body, "references": refs, "acknowledgments": acks, "captions": captions, "refs_via": refs_via}


# %% [markdown]
# ## Tier 3 — Layout-specific handlers (per-pattern)
#
# Detect Elsevier preproof / "Accepted Manuscript" PDFs and drop the metadata page 1.
# Hook for future: multi-article PDF clipping by goldset-title anchor.

# %%
ELSEVIER_PREPROOF_MARKER = "Please cite this article as"

def is_elsevier_preproof(pages: list[str]) -> bool:
    """True if page 1 contains the Elsevier accepted-manuscript boilerplate."""
    return bool(pages) and ELSEVIER_PREPROOF_MARKER.lower() in pages[0].lower()


def tier3_layout_specific(pages: list[str]) -> tuple[list[str], list[str]]:
    """Return (cleaned_pages, applied_handlers).
    Applied as a list so we can record which Tier 3 rules fired per PDF."""
    applied: list[str] = []
    if is_elsevier_preproof(pages):
        pages = pages[1:]
        applied.append("elsevier_preproof_skip_page1")
    return pages, applied


# %% [markdown]
# ## Tier 4 — Validation gate
#
# Per-PDF flags. A PDF that fires any flag goes into a manual-review queue.
# Thresholds chosen from the actual size distribution of the 235 target PDFs:
# - smallest legitimate full text was ~14k chars (3-page short paper)
# - largest was 254k chars (121-page review chapter)
# - all are >=500 chars/page (anything below that = likely scan / extraction failure)

# %%
GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda",
    "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ε": "epsilon",
}


def tier4_validate(body: str, n_pages: int, refs_found: bool, gold_title: str | None = None) -> list[str]:
    flags = []
    if n_pages > 0 and len(body) / n_pages < 500:
        flags.append("low_chars_per_page")
    if not refs_found:
        flags.append("no_references_heading")
    if gold_title:
        # Look for first 4 words of the gold title anywhere in the body, normalized.
        # Anywhere-in-body is the right check — we want to confirm we extracted the
        # right paper, not where the title appears in reading order.
        def norm(s: str) -> str:
            for g, n in GREEK_MAP.items():
                s = s.replace(g, n)
            return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())
        body_norm = norm(body)
        title_probe = " ".join(norm(gold_title).split()[:4])
        if title_probe and title_probe not in body_norm:
            flags.append("title_not_in_body")
    return flags


# %% [markdown]
# ## Run cleaning pipeline on the sample, save raw + cleaned for review
#
# Pipeline order: extract pages → Tier 3 (layout-specific) → Tier 1 (mechanical) → Tier 2 (cuts) → Tier 4 (validate)

# %%
import json

# Load goldset titles for validation
GOLDSET_PATH = Path("/Users/yun/develop/dictycite/output/dicty_gold_build/7a_dicty_gold_llm_public.jsonl")
gold_titles: dict[str, str] = {}
with GOLDSET_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        for d in rec.get("docs", []):
            pmid = str(d.get("pmid", ""))
            title = d.get("title") or ""
            if pmid and title and pmid not in gold_titles:
                gold_titles[pmid] = title

CLEAN_OUT = OUT_DIR / "cleaned"
CLEAN_OUT.mkdir(exist_ok=True)

clean_stats = []
for pmid, src_dir, label in SAMPLES:
    pdf_path = src_dir / f"{pmid}.pdf"
    if not pdf_path.exists():
        continue
    pages = extract_pages_pymupdf(pdf_path)
    raw = "\n\n".join(pages)

    # Tier 3 first — modifies page list before any text-level processing
    pages_t3, t3_applied = tier3_layout_specific(pages)
    cleaned = tier1_clean(pages_t3)
    parts = split_structural(cleaned)

    flags = tier4_validate(
        body=parts["body"],
        n_pages=len(pages_t3),
        refs_found=bool(parts["references"]),
        gold_title=gold_titles.get(pmid),
    )

    (CLEAN_OUT / f"{pmid}__01_raw.txt").write_text(raw)
    (CLEAN_OUT / f"{pmid}__02_tier1.txt").write_text(cleaned)
    (CLEAN_OUT / f"{pmid}__03_body.txt").write_text(parts["body"])
    (CLEAN_OUT / f"{pmid}__04_references.txt").write_text(parts["references"])
    (CLEAN_OUT / f"{pmid}__05_acknowledgments.txt").write_text(parts["acknowledgments"])

    clean_stats.append({
        "pmid": pmid,
        "label": label,
        "n_pages": len(pages),
        "n_pages_after_t3": len(pages_t3),
        "raw_chars": len(raw),
        "body_chars": len(parts["body"]),
        "refs_chars": len(parts["references"]),
        "ack_chars": len(parts["acknowledgments"]),
        "n_captions": len(parts["captions"]),
        "refs_found": bool(parts["references"]),
        "tier3_applied": ",".join(t3_applied) or "-",
        "tier4_flags": ",".join(flags) or "OK",
    })

clean_df = pd.DataFrame(clean_stats)
print(clean_df.to_string(index=False))

# %% [markdown]
# ## Quick sanity probes

# %%
# Show first 800 chars of body for each PDF after cleaning
for row in clean_stats:
    pmid = row["pmid"]
    print(f"\n========== {pmid} ({row['label']}) — body first 600 chars ==========")
    body = (CLEAN_OUT / f"{pmid}__03_body.txt").read_text()
    print(body[:600])
    print("...")

# %% [markdown]
# # Full-corpus pass — all 235 in-example PDFs
#
# Build target list:
# - Example union (train_200 ∪ test_50) ∩ available PDFs in `pdfs/`, minus 3 known-scans
# - Plus all PDFs in `manual/` (covers some `abstract_insufficient` PMIDs missing from `pdfs/`)
#
# Output to `output/pdf_extraction_v1/` (separate from prototype) — body only,
# plus `flag_report.jsonl` for Tier 4 review.

# %%
import os
import time

V1_OUT = Path("/Users/yun/develop/dictycite/output/pdf_extraction/v1")
V1_OUT.mkdir(exist_ok=True)
(V1_OUT / "body").mkdir(exist_ok=True)
(V1_OUT / "references").mkdir(exist_ok=True)


def all_example_pmids() -> set[str]:
    pmids = set()
    for path in [
        "/Users/yun/develop/dictycite/example/dicty_gold_llm_public_train_200.jsonl",
        "/Users/yun/develop/dictycite/example/dicty_gold_llm_public_test_50.jsonl",
    ]:
        with open(path) as f:
            for line in f:
                for d in json.loads(line).get("docs", []):
                    pmid = str(d.get("pmid", ""))
                    if pmid:
                        pmids.add(pmid)
    return pmids


KNOWN_BAD = {
    # Likely scans — extractable text < 150 chars/page (probed earlier).
    # All are abstract_supports_*; abstract from goldset suffices.
    "7771809", "7813801", "10192918",
    # Type1 CFF fonts with no ToUnicode CMap (Distiller 4.0 era); body extracts
    # as control characters. abstract_supports_detail in goldset; abstract suffices.
    "11032815",
}

example_pmids = all_example_pmids()
pdf_pmids = {f.replace(".pdf", "") for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")}
manual_pmids = {f.replace(".pdf", "") for f in os.listdir(MANUAL_DIR) if f.endswith(".pdf")}

targets: list[tuple[str, Path]] = []
for pmid in sorted(example_pmids & pdf_pmids):
    if pmid not in KNOWN_BAD:
        targets.append((pmid, PDFS_DIR / f"{pmid}.pdf"))
for pmid in sorted(manual_pmids):
    if pmid not in {p for p, _ in targets}:
        targets.append((pmid, MANUAL_DIR / f"{pmid}.pdf"))

print(f"Targets: {len(targets)} PDFs (excluded {len(KNOWN_BAD)} known-bad)")


# %% [markdown]
# ## Run cleaning on the full target list

# %%
def process_one(pmid: str, pdf_path: Path) -> dict:
    """Run the full pipeline on one PDF; return a stats row. Errors return error= field set."""
    try:
        pages = extract_pages_pymupdf(pdf_path)
        if not pages:
            return {"pmid": pmid, "error": "no_pages"}
        pages_t3, t3_applied = tier3_layout_specific(pages)
        cleaned = tier1_clean(pages_t3)
        parts = split_structural(cleaned)
        flags = tier4_validate(
            body=parts["body"],
            n_pages=len(pages_t3),
            refs_found=bool(parts["references"]),
            gold_title=gold_titles.get(pmid),
        )
        # Persist body + references (we'll chunk these next)
        (V1_OUT / "body" / f"{pmid}.txt").write_text(parts["body"])
        if parts["references"]:
            (V1_OUT / "references" / f"{pmid}.txt").write_text(parts["references"])
        return {
            "pmid": pmid,
            "src": "manual" if pdf_path.parent == MANUAL_DIR else "pdfs",
            "n_pages": len(pages),
            "n_pages_after_t3": len(pages_t3),
            "body_chars": len(parts["body"]),
            "refs_chars": len(parts["references"]),
            "ack_chars": len(parts["acknowledgments"]),
            "n_captions": len(parts["captions"]),
            "refs_found": bool(parts["references"]),
            "refs_via": parts["refs_via"],
            "tier3_applied": t3_applied,
            "tier4_flags": flags,
        }
    except Exception as e:
        return {"pmid": pmid, "src": pdf_path.parent.name, "error": f"{type(e).__name__}: {e}"}


t0 = time.time()
v1_rows: list[dict] = []
for pmid, pdf_path in targets:
    v1_rows.append(process_one(pmid, pdf_path))

elapsed = time.time() - t0
print(f"Processed {len(v1_rows)} PDFs in {elapsed:.1f}s ({elapsed/len(v1_rows)*1000:.0f} ms/pdf avg)")

# Persist the flag report
with (V1_OUT / "flag_report.jsonl").open("w") as f:
    for row in v1_rows:
        f.write(json.dumps(row) + "\n")
print(f"Wrote flag report: {V1_OUT / 'flag_report.jsonl'}")


# %% [markdown]
# ## Summarize Tier 4 flag distribution

# %%
errors = [r for r in v1_rows if "error" in r]
ok_rows = [r for r in v1_rows if "error" not in r]

print(f"\n--- Errors ({len(errors)}) ---")
for r in errors:
    print(f"  {r['pmid']} ({r.get('src','?')}): {r['error']}")

# Tier 3 hits
t3_hits = [r for r in ok_rows if r["tier3_applied"]]
print(f"\n--- Tier 3 fired ({len(t3_hits)}) ---")
for r in t3_hits:
    print(f"  {r['pmid']}: {r['tier3_applied']}")

# Tier 4 flag distribution
flag_counter = Counter()
for r in ok_rows:
    if r["tier4_flags"]:
        for fl in r["tier4_flags"]:
            flag_counter[fl] += 1
    else:
        flag_counter["OK"] += 1

print("\n--- Tier 4 flag counts ---")
for fl, c in flag_counter.most_common():
    print(f"  {fl}: {c}")

# Show flagged PDFs
flagged = [r for r in ok_rows if r["tier4_flags"]]
print(f"\n--- Flagged PDFs ({len(flagged)}) ---")
for r in flagged:
    print(f"  {r['pmid']} ({r['src']}, {r['n_pages']}p, body={r['body_chars']}): {r['tier4_flags']}")


# %% [markdown]
# ## Sanity check: body size distribution after cleaning

# %%
body_chars = sorted([r["body_chars"] for r in ok_rows])
n = len(body_chars)
if n:
    print(f"Body char count distribution (n={n}):")
    print(f"  min:    {body_chars[0]}")
    print(f"  p10:    {body_chars[int(n*0.1)]}")
    print(f"  median: {body_chars[n//2]}")
    print(f"  p90:    {body_chars[int(n*0.9)]}")
    print(f"  max:    {body_chars[-1]}")


# %% [markdown]
# # Chunking
#
# Sizes anchored on the corpus's abstract distribution (median 1,335 chars / ~250 tokens):
# - **target = 1,500 chars** (~290 tokens; just above median abstract)
# - **hard_cap = 2,500 chars** (~p95 of abstracts; almost no abstract bigger)
# - **min_size = 300 chars** (orphan tail merges into previous chunk)
# - **overlap = 200 chars** (sentence-level catch for boundary-crossing claims)
#
# Algorithm: greedy paragraph accumulator (split on `\n\n`). Captions extracted into their own chunks.
# Not sentence-aware — paragraphs are the only boundary respected.

# %%
TARGET_SIZE = 1500
HARD_CAP = 1800       # tight cap because PDF extraction merges paragraphs;
                      # without this, chunks pile up at the cap.
MIN_SIZE = 300
OVERLAP = 200


def split_captions_from_body(body: str) -> tuple[str, list[str]]:
    """Pull figure/table caption paragraphs out of the body.
    A caption paragraph is one whose FIRST line matches CAPTION_RE (e.g., 'Figure 1.', 'Table 2:').
    Returns (body_without_captions, [caption_blocks])."""
    paragraphs = body.split("\n\n")
    body_paras: list[str] = []
    captions: list[str] = []
    for p in paragraphs:
        first_line = p.lstrip().split("\n", 1)[0]
        if CAPTION_RE.match(first_line):
            captions.append(p.strip())
        else:
            body_paras.append(p)
    return "\n\n".join(body_paras), captions


def chunk_body(text: str,
               target_size: int = TARGET_SIZE,
               hard_cap: int = HARD_CAP,
               min_size: int = MIN_SIZE,
               overlap: int = OVERLAP) -> list[str]:
    """Greedy paragraph-based chunker. Not sentence-aware.
    - Accumulate paragraphs until size reaches target.
    - Single paragraph > hard_cap: split into hard_cap-sized slices.
    - Trailing chunk < min_size: merge into previous.
    - Apply overlap as a post-processing pass: each chunk after the first is prepended
      with the last `overlap` chars of the previous chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_size = 0

    def buf_text() -> str:
        return "\n\n".join(buf)

    for p in paragraphs:
        if len(p) > hard_cap:
            if buf:
                chunks.append(buf_text())
                buf, buf_size = [], 0
            for i in range(0, len(p), hard_cap):
                chunks.append(p[i:i + hard_cap])
            continue
        # +2 for the "\n\n" we'll insert when joining
        added = len(p) + (2 if buf else 0)
        if buf and buf_size + added >= target_size:
            chunks.append(buf_text())
            buf, buf_size = [], 0
        buf.append(p)
        buf_size += added

    if buf:
        leftover = buf_text()
        if chunks and len(leftover) < min_size:
            chunks[-1] = chunks[-1] + "\n\n" + leftover
        else:
            chunks.append(leftover)

    # Apply overlap as post-pass
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            out.append(tail + "\n\n" + chunks[i])
        chunks = out

    return chunks


# %% [markdown]
# ## Quick test on a sample paper

# %%
test_pmid = "29626371"  # 16-page modern paper, well-extracted
body = (V1_OUT / "body" / f"{test_pmid}.txt").read_text()
body_no_caps, caps = split_captions_from_body(body)
chunks = chunk_body(body_no_caps)
print(f"=== {test_pmid} ===")
print(f"body chars (with caps): {len(body)}")
print(f"body chars (no caps):   {len(body_no_caps)}")
print(f"captions extracted:     {len(caps)} (total chars: {sum(len(c) for c in caps)})")
print(f"body chunks:            {len(chunks)}")
print(f"chunk sizes:            {[len(c) for c in chunks]}")
print()
print("--- chunk 0 (first 200 chars) ---")
print(chunks[0][:200])
print("\n--- chunk 1 (first 200 chars; should start with overlap from chunk 0) ---")
print(chunks[1][:200])

# %% [markdown]
# ## Run chunker on all 234 cleaned bodies, save chunks.jsonl

# %%
CHUNKS_OUT = V1_OUT / "chunks.jsonl"

chunk_rows: list[dict] = []
with (V1_OUT / "flag_report.jsonl").open() as f:
    cleaned_records = [json.loads(line) for line in f]

for rec in cleaned_records:
    if "error" in rec:
        continue
    pmid = rec["pmid"]
    body_path = V1_OUT / "body" / f"{pmid}.txt"
    if not body_path.exists():
        continue
    body = body_path.read_text()
    body_no_caps, caps = split_captions_from_body(body)
    body_chunks = chunk_body(body_no_caps)
    body_total = len(body_no_caps) or 1

    # Body chunks
    cum = 0
    for i, ck in enumerate(body_chunks, start=1):
        chunk_rows.append({
            "pmid": pmid,
            "chunk_id": f"{pmid}#body_{i:03d}",
            "type": "body",
            "seq": i,
            "text": ck,
            "n_chars": len(ck),
            "position_frac": round(cum / body_total, 3),
        })
        cum += len(ck) - OVERLAP  # approximate

    # Caption chunks
    for i, cap in enumerate(caps, start=1):
        chunk_rows.append({
            "pmid": pmid,
            "chunk_id": f"{pmid}#caption_{i:03d}",
            "type": "caption",
            "seq": i,
            "text": cap,
            "n_chars": len(cap),
            "position_frac": None,
        })

# Note: abstract chunks are NOT generated here — they come from the existing corpus
# (gold metadata). They get merged in at the corpus-assembly step (next phase).

with CHUNKS_OUT.open("w") as f:
    for row in chunk_rows:
        f.write(json.dumps(row) + "\n")
print(f"Wrote {len(chunk_rows)} chunks to {CHUNKS_OUT}")

# %% [markdown]
# ## Chunk distribution stats

# %%
from collections import defaultdict
by_pmid = defaultdict(lambda: {"body": 0, "caption": 0})
sizes_body, sizes_cap = [], []
for r in chunk_rows:
    by_pmid[r["pmid"]][r["type"]] += 1
    if r["type"] == "body":
        sizes_body.append(r["n_chars"])
    else:
        sizes_cap.append(r["n_chars"])

print(f"PMIDs with at least one full-text chunk: {len(by_pmid)}")
print(f"Total body chunks:    {len(sizes_body)}")
print(f"Total caption chunks: {len(sizes_cap)}")

print("\n--- Body chunk size distribution ---")
sb = sorted(sizes_body)
n = len(sb)
for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
    print(f"  p{int(q*100):>2}: {sb[int(n*q)]}")
print(f"  max: {sb[-1]}")

print("\n--- Body chunks per PMID distribution ---")
counts = sorted([v["body"] for v in by_pmid.values()])
for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
    print(f"  p{int(q*100):>2}: {counts[int(len(counts)*q)]}")
print(f"  max: {counts[-1]}")

print("\n--- Caption chunks per PMID (top 10) ---")
top_caps = sorted(by_pmid.items(), key=lambda x: -x[1]["caption"])[:10]
for pmid, c in top_caps:
    print(f"  {pmid}: {c['caption']} captions")

# %%

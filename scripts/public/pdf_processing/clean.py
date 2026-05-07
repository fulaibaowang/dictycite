"""Tier 1–4 cleaning passes applied to raw PyMuPDF text."""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Tier 1 — mechanical cleaning
# ---------------------------------------------------------------------------

LIGATURE_MAP = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬀ": "ff",
    "ﬅ": "ft", "ﬆ": "st",
    "­": "",   # soft hyphen
}


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


def clean_chars(text: str) -> str:
    for k, v in LIGATURE_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"\(cid:\d+\)", "", text)
    return text


def dehyphenate_linebreaks(text: str) -> str:
    """Join 'co-\\nword' → 'coword' when the prefix doesn't look like a real compound."""
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


_NOISE_PATTERNS = [
    r"^doi[:\s]\s*10\.\d",                                  # DOI line
    r"^\d{4}-\d{4}/\d+/?\$",                                # ISSN/$ pattern
    r"copyright\s+©|©\s*\d{4}|all\s+rights\s+reserved",     # copyright (©)
    r"^[©0]\s*\d{4}\s+by\s+the",                            # OCR-mangled "© 1993 by The..."
    r"^vol\.?\s*\d|^vol\s+\d",                              # Vol. N
    r"^pp\.\s*\d|^p\.\s*\d+\s*[-–]\s*\d+\s*$",              # pp. N-N
    r"^no\.\s*\d+\s*$",                                     # No. N
    r"^issue\s+of\s+",                                      # "Issue of ..."
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4},?\s+p\.?\s*\d",
    r"first\s+published\s+online",
    r"^printed\s+in\s+",
    r"^\(?received\s+(for\s+publication|on|in|by)|^\(?received\s+\d",
    r"^submitted\s+\w+\s+\d+",
    r"^accepted\s+\w+\s+\d+|^accepted\s+for\s+",
    r"^revised\s+(form|\w+\s+\d+)",
    r"^published\s+(in\s+press|online|by)",
    r"^communicated\s+by",
    r"^edited\s+by",
    r"^\*?\s*for\s+correspondence",                          # "*For correspondence"
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


# ---------------------------------------------------------------------------
# Tier 2 — structural cuts (References, Acknowledgments, Captions)
# ---------------------------------------------------------------------------

# References heading: line on its own, case-insensitive, with optional numbering.
# Variants in the corpus: "References", "References and Notes" (Science),
# "References Cited", "Reference List", "Bibliography", "Literature Cited", "Works Cited".
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
    num_line_re = re.compile(r"^\s*(\d{1,3})[.\s]\s*[A-Z]", re.MULTILINE)
    matches = list(num_line_re.finditer(text))
    if len(matches) < min_consecutive:
        return None
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
        run_start = _detect_unlabeled_refs_run(text)
        if run_start is not None:
            cut_at = min(cut_at, run_start)
            refs = text[run_start:].strip()
            refs_via = "unlabeled_run"
    if ack_m and ack_m.start() < cut_at:
        ack_end = refs_m.start() if refs_m else len(text)
        if not refs_m or (ack_end - ack_m.start()) < 0.5 * len(text):
            acks = text[ack_m.end():ack_end].strip()
            cut_at = min(cut_at, ack_m.start())
    body = text[:cut_at].strip()

    captions = [m.group(0).strip() for m in CAPTION_RE.finditer(body)]
    return {"body": body, "references": refs, "acknowledgments": acks, "captions": captions, "refs_via": refs_via}


# ---------------------------------------------------------------------------
# Tier 3 — layout-specific handlers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tier 4 — validation gate
# ---------------------------------------------------------------------------

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
        def norm(s: str) -> str:
            for g, n in GREEK_MAP.items():
                s = s.replace(g, n)
            return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())
        body_norm = norm(body)
        title_probe = " ".join(norm(gold_title).split()[:4])
        if title_probe and title_probe not in body_norm:
            flags.append("title_not_in_body")
    return flags

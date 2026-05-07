"""Shared constants for the PDF processing pipeline."""

from __future__ import annotations

from pathlib import Path

# Default output root (override with --out on each CLI).
DEFAULT_OUT_DIR = Path("output/pdf_extraction/v1")

# PMIDs whose PDFs cannot be cleanly extracted. All four are abstract_supports_*
# in the goldset, so the curated abstract from PubMed is used instead.
#   7771809, 7813801, 10192918  — page-image scans (no extractable text)
#   11032815                    — Type1 CFF fonts, no ToUnicode CMap
KNOWN_BAD: frozenset[str] = frozenset({
    "7771809",
    "7813801",
    "10192918",
    "11032815",
})

# Chunking parameters (anchored on abstract median of 1,335 chars).
CHUNK_TARGET = 1500
CHUNK_CAP = 1800
CHUNK_OVERLAP = 200
CHUNK_MIN = 300

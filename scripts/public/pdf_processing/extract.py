"""PDF text extraction wrappers (PyMuPDF)."""

from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf


def extract_pages_pymupdf(pdf_path: Path) -> list[str]:
    """Extract text per page in raw PDF stream order.

    Two-column layouts are handled correctly by the underlying stream order on
    well-formed papers. Returns one string per page; empty pages are kept so
    repeated header/footer detection can use page indices.
    """
    with fitz.open(pdf_path) as doc:
        return [page.get_text() for page in doc]

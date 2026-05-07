"""Caption splitting and body chunking."""

from __future__ import annotations

from .clean import CAPTION_RE
from .config import CHUNK_CAP, CHUNK_MIN, CHUNK_OVERLAP, CHUNK_TARGET


def split_captions_from_body(body: str) -> tuple[str, list[str]]:
    """Pull figure/table caption paragraphs out of the body.
    A caption paragraph is one whose FIRST line matches CAPTION_RE
    (e.g., 'Figure 1.', 'Table 2:'). Returns (body_without_captions, [caption_blocks]).
    """
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
               target_size: int = CHUNK_TARGET,
               hard_cap: int = CHUNK_CAP,
               min_size: int = CHUNK_MIN,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
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

    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            out.append(tail + "\n\n" + chunks[i])
        chunks = out

    return chunks

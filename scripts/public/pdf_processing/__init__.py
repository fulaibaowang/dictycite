"""PDF extraction & chunking pipeline for the DictyCite RAG corpus.

Stages:
    fetch_titles.py  PMID → PubMed title (cached as titles.json)
    clean_pdfs.py    PDF → cleaned body / references / flag report
    chunk_bodies.py  cleaned body → chunks.jsonl

See output/pdf_extraction/STATUS.md for engineering notes and decisions.
"""

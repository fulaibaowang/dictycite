import requests
import xml.etree.ElementTree as ET
from typing import Tuple, Optional
import time


def pmid_to_apa(pmid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a PubMed ID (PMID), fetch metadata from NCBI and return APA citations.

    Implements retries with exponential backoff on 5xx/network errors.

    Returns (full, short) on success, or (None, None) on failure.
    """
    # Fetch article data from NCBI
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}

    resp = None
    attempts = 3
    backoff = 1
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"Warning: failed to fetch metadata for PMID {pmid}: {e}")
            return (None, None)

    if resp is None:
        return (None, None)

    try:
        root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"Warning: could not parse XML for PMID {pmid}: {e}")
        return (None, None)

    article = root.find(".//PubmedArticle")
    if article is None:
        return (None, None)

    # Extract fields
    article_title = article.findtext(".//ArticleTitle")
    journal_title = article.findtext(".//Journal/Title")
    year = article.findtext(".//PubDate/Year") or "n.d."
    volume = article.findtext(".//JournalIssue/Volume")
    issue = article.findtext(".//JournalIssue/Issue")
    pages = article.findtext(".//Pagination/MedlinePgn")
    doi = article.findtext(".//ArticleId[@IdType='doi']")
    authors = []

    for author in article.findall(".//Author"):
        last = author.findtext("LastName")
        initials = author.findtext("Initials")
        if last and initials:
            authors.append(f"{last}, {initials}.")

    # ---- Format author list for reference ----
    if len(authors) == 0:
        author_str = ""
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) <= 7:
        author_str = ", ".join(authors[:-1]) + ", & " + authors[-1]
    else:
        author_str = ", ".join(authors[:6]) + ", ... " + authors[-1]

    # ---- Build APA reference citation ----
    full = f"{author_str} ({year}). {article_title}. *{journal_title}*, {volume}"
    if issue:
        full += f"({issue})"
    if pages:
        full += f", {pages}"
    if doi:
        full += f". https://doi.org/{doi}"
    else:
        full += "."

    # ---- Build short in-text citation ----
    if len(authors) == 0:
        short = f"({journal_title}, {year})"
    elif len(authors) == 1:
        short = f"({authors[0].split(',')[0]}, {year})"
    elif len(authors) == 2:
        short = f"({authors[0].split(',')[0]} & {authors[1].split(',')[0]}, {year})"
    else:
        short = f"({authors[0].split(',')[0]} et al., {year})"

    return full, short

import requests
import xml.etree.ElementTree as ET
from typing import Tuple


def pmid_to_apa(pmid: str) -> Tuple[str, str]:
    """
    Given a PubMed ID (PMID), fetch metadata from NCBI and return APA citations.

    Args:
        pmid: PubMed ID as string

    Returns:
        Tuple of (full_citation, short_citation):
        - full: APA-style reference list citation
        - short: in-text APA citation (e.g., "Smith et al., 2020")
    """
    # Fetch article data from NCBI
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    response = requests.get(url, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    article = root.find(".//PubmedArticle")
    if article is None:
        return ("Article not found.", "Article not found.")

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

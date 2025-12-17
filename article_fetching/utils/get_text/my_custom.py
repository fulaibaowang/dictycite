import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

USER_AGENT = "pmcid-sections/1.0 (you@example.com)"
TIMEOUT = 45


# ---------------- HTTP ----------------
def _get(url: str, params: dict | None = None) -> str:
    r = requests.get(
        url, params=params or {}, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.text


# ---------------- Text utils ----------------
def _txt(el: Optional[ET.Element]) -> str:
    """Flatten text of an element and its descendants, collapsing whitespace."""
    if el is None:
        return ""
    parts: List[str] = []

    def walk(n: ET.Element):
        if n.text:
            parts.append(n.text)
        for c in list(n):
            walk(c)
            if c.tail:
                parts.append(c.tail)

    walk(el)
    return " ".join(" ".join(parts).split())


def _is(el: ET.Element, name: str) -> bool:
    return el.tag.endswith(name)


# ---------------- JATS parsing ----------------
def _first_article(root: ET.Element) -> Optional[ET.Element]:
    if _is(root, "article"):
        return root
    return root.find(".//article")


def _collect_level_paragraphs(container: ET.Element) -> List[str]:
    """
    Collect paragraph-like text at the *current* level (exclude nested <sec>).
    Each returned string is ONE paragraph.
    """
    out: List[str] = []
    for node in list(container):
        if _is(node, "title") or _is(node, "label") or _is(node, "sec"):
            continue
        # treat common blocks as paragraphs; fallback captures any non-sec text block
        if any(
            _is(node, t)
            for t in (
                "p",
                "boxed-text",
                "disp-quote",
                "def-list",
                "list",
                "supplementary-material",
                "table-wrap-foot",
                "statement",
                "caption",
            )
        ):
            txt = _txt(node).strip()
            if txt:
                out.append(txt)
        else:
            txt = _txt(node).strip()
            if txt:
                out.append(txt)
    return out


def _parse_body_into_map(article: ET.Element, store: Dict[str, List[str]]):
    body = article.find("./body")
    if body is None:
        return

    # any preface text before first <sec> -> one "Body (untitled)" section
    preface: List[str] = []
    for node in list(body):
        if _is(node, "sec"):
            break
        if not _is(node, "title"):
            t = _txt(node).strip()
            if t:
                preface.append(t)
    if preface:
        store.setdefault("Body (untitled)", []).extend(preface)

    def walk_sec(sec: ET.Element):
        title = _txt(sec.find("./title")) or "Untitled Section"
        paras = _collect_level_paragraphs(sec)  # already one string per paragraph
        if paras:
            store.setdefault(title, []).extend(paras)
        # recurse into subsections; each gets its own key (by its own title)
        for child in list(sec):
            if _is(child, "sec"):
                walk_sec(child)

    for top in body.findall("./sec"):
        walk_sec(top)


def _parse_abstract_into_map(article: ET.Element, store: Dict[str, List[str]]):
    meta = article.find("./front/article-meta")
    if meta is None:
        return

    abs_nodes = meta.findall("./abstract") + meta.findall("./trans-abstract")
    if not abs_nodes:
        return

    base = "Abstract"
    for i, abs_el in enumerate(abs_nodes, 1):
        title_text = _txt(abs_el.find("./title"))
        key = (
            base
            if (len(abs_nodes) == 1 and not title_text)
            else (title_text or f"{base} {i}")
        )

        # paragraphs directly under <abstract>
        paras = _collect_level_paragraphs(abs_el)  # one string per paragraph
        if paras:
            store.setdefault(key, []).extend(paras)

        # nested abstract subsections get their own keys
        for s in [c for c in list(abs_el) if _is(c, "sec")]:
            stitle = _txt(s.find("./title")) or f"{key} subsection"
            s_paras = _collect_level_paragraphs(s)
            if s_paras:
                store.setdefault(stitle, []).extend(s_paras)


# ---------------- Public API ----------------
def get_ncbi_text_my(pmcid: str) -> Dict[str, List[str]]:
    """
    Return {"Title":[<title>], "Abstract":[paragraphs], "<Section>":[paragraphs], ...}
    using NCBI PMC efetch (db=pmc).
    """
    xml = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "retmode": "xml"},
    )
    return _jats_to_section_paragraph_map(xml)


def get_epmc_text_my(pmcid: str) -> Dict[str, List[str]]:
    """
    Return {"Title":[<title>], "Abstract":[paragraphs], "<Section>":[paragraphs], ...}
    using Europe PMC fullTextXML.
    """
    xml = _get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    return _jats_to_section_paragraph_map(xml)


def _jats_to_section_paragraph_map(raw_xml: str) -> Dict[str, List[str]]:
    """
    JATS XML -> {section_title: [paragraphs]}.
    - "Title" -> single-element list with the article title string.
    - "Abstract" -> list of paragraphs (handles multiple abstracts / subsections).
    - Body <sec> and nested <sec> are each separate keys by their own titles.
    """
    out: Dict[str, List[str]] = {}
    root = ET.fromstring(raw_xml)
    article = _first_article(root)
    if article is None:
        return out

    # Title
    title_el = article.find("./front/article-meta/title-group/article-title")
    if title_el is None:
        title_el = article.find(
            "./front/article-meta/title-group/trans-title-group/trans-title"
        )
    title = _txt(title_el)
    if title:
        out["Title"] = [title]

    # Abstract(s)
    _parse_abstract_into_map(article, out)

    # Body sections (+ subsections)
    _parse_body_into_map(article, out)

    return out

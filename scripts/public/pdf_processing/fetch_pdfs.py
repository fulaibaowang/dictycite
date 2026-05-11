#!/usr/bin/env python3
"""Fetch PDFs for a list of PMIDs and save them to a target directory.

Usage:
    python -m scripts.public.pdf_processing.fetch_pdfs \\
        --pmids-file output/pdf_extraction/missing_pmids_7a.txt \\
        --out-dir /Users/yun/Documents/dictybase_papers/manual \\
        --skip-dir /Users/yun/Documents/dictybase_papers/pdfs \\
        --skip-dir /Users/yun/Documents/dictybase_papers/manual \\
        --report output/pdf_extraction/fetch_pdfs_report.json

PMIDs already present in any --skip-dir are not refetched. Output filename is
``<pmid>.pdf``.

Strategy (per PMID, stop at first success):
    1. PMID -> PMCID via NCBI idconv.
    2. If PMCID found, try Europe PMC render URL
       (``https://europepmc.org/articles/PMC<id>?pdf=render``).
    3. Fallback: PMC Open-Access service. Calls ``oa.fcgi`` to retrieve the
       PDF link (FTP -> HTTPS); downloads if present. Only works for OA papers.

Failures are recorded with the reason. Set ``NCBI_API_KEY`` to raise NCBI's
rate limit from 3 req/s to 10 req/s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EPMC_RENDER_URL = "https://europepmc.org/articles/PMC{pmcid}?pdf=render"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

USER_AGENT = "dictycite-pdf-processing/1.0 (mailto:alterchen.wy@gmail.com)"
HTTP_TIMEOUT = 60
PDF_MIN_BYTES = 5000  # anything smaller is almost certainly an error page
DELAY_NO_KEY = 0.34   # ~3 req/s
DELAY_WITH_KEY = 0.11 # ~10 req/s


@dataclass
class Result:
    pmid: str
    status: str        # "downloaded" | "already_have" | "no_pmcid" | "not_oa" | "http_error" | "too_small" | "exception"
    pmcid: str | None = None
    source: str | None = None  # "epmc_render" | "pmc_oa"
    bytes: int | None = None
    detail: str | None = None


def read_pmids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s.isdigit():
            out.append(s)
    return out


def already_have(pmid: str, skip_dirs: list[Path]) -> bool:
    return any((d / f"{pmid}.pdf").exists() for d in skip_dirs)


def http_get(url: str, accept: str | None = None) -> tuple[int, bytes, str]:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read()
            return resp.status, body, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, b"", ""
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, b"", str(e)


def pmid_to_pmcid(pmid: str, api_key: str | None) -> str | None:
    params = {"ids": pmid, "format": "json", "tool": "dictycite", "email": "alterchen.wy@gmail.com"}
    if api_key:
        params["api_key"] = api_key
    url = f"{IDCONV_URL}?{urllib.parse.urlencode(params)}"
    status, body, _ = http_get(url, accept="application/json")
    if status != 200 or not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    for rec in data.get("records", []):
        pmcid = rec.get("pmcid")
        if pmcid:
            return pmcid.replace("PMC", "")
    return None


def is_pdf(body: bytes, content_type: str) -> bool:
    if body[:4] == b"%PDF":
        return True
    return "application/pdf" in content_type.lower()


def fetch_epmc_render(pmcid: str) -> tuple[bytes | None, str]:
    url = EPMC_RENDER_URL.format(pmcid=pmcid)
    status, body, ct = http_get(url, accept="application/pdf")
    if status != 200 or not body:
        return None, f"http_{status}"
    if not is_pdf(body, ct):
        return None, f"not_pdf (ct={ct[:40]})"
    if len(body) < PDF_MIN_BYTES:
        return None, f"too_small ({len(body)} bytes)"
    return body, "ok"


def fetch_pmc_oa(pmcid: str, api_key: str | None) -> tuple[bytes | None, str]:
    params = {"id": f"PMC{pmcid}"}
    if api_key:
        params["api_key"] = api_key
    url = f"{PMC_OA_URL}?{urllib.parse.urlencode(params)}"
    status, body, _ = http_get(url)
    if status != 200 or not body:
        return None, f"oa_http_{status}"
    xml = body.decode("utf-8", errors="replace")
    # OA service returns an idList with <link format="pdf" href="..."/>
    m = re.search(r'<link\s+[^>]*format="pdf"[^>]*href="([^"]+)"', xml)
    if not m:
        if "is not Open Access" in xml or "idDoesNotExist" in xml:
            return None, "not_oa"
        return None, "no_pdf_link_in_oa_response"
    pdf_url = m.group(1).replace("ftp://", "https://")
    status, pdf_body, ct = http_get(pdf_url, accept="application/pdf")
    if status != 200 or not pdf_body:
        return None, f"oa_dl_http_{status}"
    # Many OA "pdf" links return a tarball — only accept actual PDFs.
    if not is_pdf(pdf_body, ct):
        return None, f"oa_not_pdf (ct={ct[:40]}, url={pdf_url[-30:]})"
    if len(pdf_body) < PDF_MIN_BYTES:
        return None, f"too_small ({len(pdf_body)} bytes)"
    return pdf_body, "ok"


def fetch_one(pmid: str, out_dir: Path, api_key: str | None) -> Result:
    pmcid = pmid_to_pmcid(pmid, api_key)
    if not pmcid:
        return Result(pmid=pmid, status="no_pmcid")

    pdf, detail = fetch_epmc_render(pmcid)
    if pdf:
        (out_dir / f"{pmid}.pdf").write_bytes(pdf)
        return Result(pmid=pmid, status="downloaded", pmcid=pmcid, source="epmc_render", bytes=len(pdf))

    pdf, detail2 = fetch_pmc_oa(pmcid, api_key)
    if pdf:
        (out_dir / f"{pmid}.pdf").write_bytes(pdf)
        return Result(pmid=pmid, status="downloaded", pmcid=pmcid, source="pmc_oa", bytes=len(pdf))

    if detail2 == "not_oa":
        return Result(pmid=pmid, status="not_oa", pmcid=pmcid, detail=f"epmc={detail}; oa=not_oa")
    return Result(pmid=pmid, status="http_error", pmcid=pmcid, detail=f"epmc={detail}; oa={detail2}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmids-file", type=Path, required=True,
                    help="File with one PMID per line (blank lines and # comments ignored).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Users/yun/Documents/dictybase_papers/manual"),
                    help="Where to save downloaded PDFs (default: manual/).")
    ap.add_argument("--skip-dir", type=Path, action="append", default=[],
                    help="Directory whose <pmid>.pdf files are considered already-have (repeatable).")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path to write a JSON report of per-PMID outcomes.")
    ap.add_argument("--max", type=int, default=None,
                    help="Stop after this many fetch attempts (for smoke testing).")
    args = ap.parse_args()

    pmids = read_pmids(args.pmids_file)
    if not pmids:
        print("No PMIDs in input file.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Always treat the out-dir as a skip-dir too, to support resume-on-rerun.
    skip_dirs = list(dict.fromkeys([args.out_dir, *args.skip_dir]))

    api_key = os.environ.get("NCBI_API_KEY")
    delay = DELAY_WITH_KEY if api_key else DELAY_NO_KEY

    todo = [pm for pm in pmids if not already_have(pm, skip_dirs)]
    skipped = len(pmids) - len(todo)
    if args.max is not None:
        todo = todo[: args.max]

    print(f"Input PMIDs           : {len(pmids)}")
    print(f"Already have (skipped): {skipped}")
    print(f"Will attempt          : {len(todo)}")
    print(f"Out dir               : {args.out_dir}")
    print(f"NCBI API key          : {'yes' if api_key else 'no (3 req/s)'}")
    print()

    results: list[Result] = []
    counts = {"downloaded": 0, "no_pmcid": 0, "not_oa": 0, "http_error": 0, "exception": 0}
    for i, pmid in enumerate(todo, 1):
        try:
            r = fetch_one(pmid, args.out_dir, api_key)
        except Exception as e:  # noqa: BLE001 — best-effort batch
            r = Result(pmid=pmid, status="exception", detail=str(e))
        results.append(r)
        counts[r.status] = counts.get(r.status, 0) + 1
        marker = {
            "downloaded": "OK",
            "no_pmcid":   "--",
            "not_oa":     "!!",
            "http_error": "XX",
            "exception":  "EE",
            "too_small":  "??",
        }.get(r.status, "??")
        bytes_s = f" {r.bytes:>7}b" if r.bytes else ""
        src_s = f" [{r.source}]" if r.source else ""
        det_s = f"  {r.detail}" if r.detail and r.status != "downloaded" else ""
        print(f"[{i:>3}/{len(todo)}] {marker} {pmid} pmcid={r.pmcid or '-':<8}{src_s}{bytes_s}{det_s}")
        if i < len(todo):
            time.sleep(delay)

    print()
    print("Summary:")
    for k, v in counts.items():
        print(f"  {k:<12} {v}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nReport: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

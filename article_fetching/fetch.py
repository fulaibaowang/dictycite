import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from datetime import datetime

from utils.citation import pmid_to_apa
from utils.get_text import (
    get_epmc_text,
    get_epmc_text_my,
    get_ncbi_text,
    get_ncbi_text_my,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Search query.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to fetch. It fetches all by default.",
    )
    parser.add_argument(
        "--get_text_from",
        type=str,
        default=None,
        choices=["epmc", "epmc_my", "ncbi", "ncbi_my"],
        help="Where to fetch the full text from. It fetches no text by default.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Save folder for output. It creates a folder with datetimestamp in the current directory by default.",
    )

    return parser.parse_args()


def get_max_records(query: str, base_url: str) -> int:
    """
    Pre-query to get total number of hits for a given query.

    Args:
        query: Search query string
        base_url: Europe PMC API base URL

    Returns:
        Total number of hits

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the response cannot be parsed
    """
    params = {
        "query": query,
        "format": "json",
        "cursorMark": "*",
        "pageSize": 1,  # Europe PMC requires at least 1 for resultType=core
        "resultType": "core",
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        total_hits = data.get("hitCount", 0)
        return total_hits
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to get max records: {e}") from e


def fetch(
    query,
    output_dir,
    max_records=None,
    get_text_from=None,
):
    """
    Fetches article metadata and optionally full texts from Europe PMC and saves them to JSON files.

    Args:
        query: Search query string for Europe PMC API
        output_dir: Directory path where JSON files will be saved
        max_records: Maximum number of records to fetch (None = fetch all)
        get_text_from: Source for full text fetching ("epmc", "epmc_my", "ncbi", "ncbi_my", or None)
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"  # Europe PMC's REST API
    cursor = "*"
    records_fetched = 0

    # ensure output directory exists (handle when user passes --output_path)
    os.makedirs(output_dir, exist_ok=True)

    if max_records is None:
        max_records = get_max_records(query, base_url)

    pbar = tqdm(total=max_records, desc="Fetching articles")  # progress bar

    while True:
        params = {
            "query": query,
            "format": "json",
            "cursorMark": cursor,
            "pageSize": 1000,  # Europe PMC limit
            "resultType": "core",
        }
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
        except requests.RequestException as e:
            print(f"Error fetching batch: {e}")
            break

        for record in results:
            epmc_id = record.get("id", None)  # Europe PMC ID
            pmid = record.get("pmid", None)  # PubMed ID
            pmcid = record.get("pmcid", None)  # PubMed Central ID

            # if pmid is None:
            #     print(f"Skip: No PMID for record {epmc_id}")
            #     continue  # skip articles without PMID
            if pmid is None:
                # not an error; many PMC records have no PMID
                pass

            # choose a stable id for filename
            file_id = pmcid or pmid or epmc_id
            if file_id is None:
                print("Skip: No identifier (pmcid/pmid/id).")
                continue

            # get journal
            try:
                journal = record.get("journalInfo", {}).get("journal", {}).get("title")
            except (AttributeError, TypeError):
                journal = None

            # get citation
            # pmid can be None for many PMC records; guard the ncbi request
            apa_full, apa_short = (None, None)
            if pmid:
                try:
                    apa_full, apa_short = pmid_to_apa(pmid)
                except Exception as e:
                    print(f"Warning: failed to fetch citation for PMID {pmid}: {e}")

            output_record = {
                "id": epmc_id,
                "pmid": pmid,
                "pmcid": pmcid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                # "urlPDF": None,
                "title": record.get("title", None),
                "authors": record.get("authorString", None),
                "journal": journal,
                "year": record.get("pubYear", None),
                "doi": record.get("doi", None),
                "license": record.get("license", None),
                "citation": {
                    "apa": apa_full,
                    "apa_short": apa_short,
                },
                "abstract": record.get("abstractText", None),
                "text": None,
            }

            # fetch full text
            if get_text_from is not None and pmcid is not None:
                if get_text_from == "ncbi":
                    output_record["text"] = get_ncbi_text(pmcid)
                elif get_text_from == "epmc":
                    output_record["text"] = get_epmc_text(pmcid)
                elif get_text_from == "epmc_my":
                    output_record["text"] = get_epmc_text_my(pmcid)
                elif get_text_from == "ncbi_my":
                    output_record["text"] = get_ncbi_text_my(pmcid)

            # if fetch_pdf:
            #    for ft in record.get("fullTextUrlList", {}).get("fullTextUrl", []):
            #        url_pdf = ft.get("url")
            #        if ft.get("documentStyle") == "pdf":
            #            output_record["urlPDF"] = url_pdf

            # Determine filename
            # filename = pmid + ".json"
            filename = f"{file_id}.json"

            output_path = os.path.join(output_dir, filename)

            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(output_record, out_f, ensure_ascii=False, indent=2)

            records_fetched += 1
            pbar.update(1)

            if max_records is not None and records_fetched >= max_records:
                break

        cursor = data.get("nextCursorMark")
        if not cursor:
            print("No next cursor found. Finished fetching.")
            break

        time.sleep(0.25)

    pbar.close()

    print(f"\nDone! Saved {records_fetched} individual JSON files in '{output_dir}'.")


def main():

    args = parse_args()

    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_path = f"fetch-{timestamp}"
        os.makedirs(args.output_path, exist_ok=True)

    fetch(
        query=args.query,
        output_dir=args.output_path,
        max_records=args.max_records,
        get_text_from=args.get_text_from,
    )


if __name__ == "__main__":
    main()

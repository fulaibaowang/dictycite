import requests


def get_ncbi_text(pmcid):
    """
    Fetches full text from NCBI BioC API.

    Returns:
        Dict: {'section1': [paragraph1, ...], ...}
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            if not resp.text.startswith("[Error]"):
                bioc = resp.json()
                if isinstance(bioc, list):
                    bioc = bioc[0]  # sometimes bioc is a list of length 1
                passages = []
                # print("bioc4")
                for doc in bioc.get("documents", []):
                    for passage in doc.get("passages", []):
                        passages.append(
                            {
                                "section_type": passage.get("infons", {}).get(
                                    "section_type"
                                ),
                                "text": passage.get("text"),
                            }
                        )
                return passages
    except Exception as e:
        print(f"Error NCBI BioC Fetch {pmcid}: {e}")
    return None

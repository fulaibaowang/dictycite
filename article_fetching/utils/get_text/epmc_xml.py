import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Suppress R console output messages
import rpy2.rinterface_lib.callbacks as rcb

rcb.consolewrite_print = lambda x: None
rcb.consolewrite_warnerror = lambda x: None

TIDYPMC = importr("tidypmc")  # Import R package tidypmc for XML extraction


def get_epmc_text(pmcid, tidypmc=TIDYPMC):
    """
    Fetches fullTextXML from Europe PMC and converts it to structured text.

    Returns:
        Dict: {'section1': [paragraph1, ...], ...}
    """

    try:
        # Download the article XML

        doc = tidypmc.pmc_xml(pmcid)

        # Extract plain text
        text_df = tidypmc.pmc_text(doc)

        # Convert to pandas DataFrame using localconverter
        with localconverter(ro.default_converter + pandas2ri.converter):
            text_pd = ro.conversion.rpy2py(text_df)

        # Convert to dict: section -> list of sentences
        text_dict = {}

        for _, row in text_pd.iterrows():
            section = row["section"]
            paragraph_idx = row["paragraph"]
            sentence = row["text"]

            if section not in text_dict:
                text_dict[section] = {}

            if paragraph_idx not in text_dict[section]:
                text_dict[section][paragraph_idx] = []

            text_dict[section][paragraph_idx].append(sentence)

        # Now join the sentences per paragraph and flatten the dict
        output_dict = {}

        for section, paragraphs in text_dict.items():
            output_dict[section] = []
            for idx in sorted(paragraphs.keys()):
                paragraph = " ".join(paragraphs[idx])
                output_dict[section].append(paragraph)

        return output_dict

    except Exception as e:
        print(f"Error EPMC XML Fetch {pmcid}: {e}")

    return None

import os
import json
import argparse
import logging
from collections import Counter
from typing import Optional


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the directory containing JSON files.",
    )
    return parser.parse_args()


def analyze_json_files(directory: str) -> None:
    """
    Analyze JSON files in the specified directory for text content and license distribution.

    Args:
        directory (str): Path to the directory containing JSON files.
        verbose (bool): If True, enable detailed logging output.

    Returns:
        None
    """

    total_articles = 0
    abstract_present = 0
    text_present = 0
    license_counter = Counter()

    # Counters for text content analysis
    # only_title = 0
    # only_abstract = 0
    # title_and_abstract = 0
    # more_than_title_abstract = 0

    if not os.path.isdir(directory):
        print(f"The provided path is not a directory: {directory}")
        return

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            print(f"Skipping file '{filename}': {e}")
            continue

        total_articles += 1

        # Count abstract presence
        abstract = data.get("abstract")
        if abstract is not None:
            abstract_present += 1

        # Count text presence
        text = data.get("text")
        if text is not None:
            text_present += 1

            # if isinstance(textxml, dict):
            #    keys = set(textxml.keys())
            #    if keys == {"Title"}:
            #        only_title += 1
            #    elif keys == {"Abstract"}:
            #        only_abstract += 1
            #    elif keys == {"Title", "Abstract"}:
            #        title_and_abstract += 1
            #    else:
            #        more_than_title_abstract += 1

        # Count license occurrences
        license_value: Optional[str] = data.get("license")
        license_counter[license_value] += 1

    if total_articles == 0:
        logging.warning("No valid JSON files found.")
        return

    # Calculate and print TextXML percentage
    text_percentage = (text_present / total_articles) * 100
    abstract_percentage = (abstract_present / total_articles) * 100
    print(f"Total articles processed: {total_articles}")
    print(f"Percentage of articles with full text: {text_percentage:.2f}%")
    print(f"Percentage of articles with abstract: {abstract_percentage:.2f}%")

    # TextXML content analysis
    # print("TextXML Content Analysis (only for articles with TextXML):")
    # if textxml_present > 0:
    #    print(
    #        f"Only 'Title': {only_title} ({(only_title / textxml_present) * 100:.2f}%)"
    #    )
    #    print(
    #        f"Only 'Abstract': {only_abstract} ({(only_abstract / textxml_present) * 100:.2f}%)"
    #    )
    #    print(
    #        f"'Title' and 'Abstract' only: {title_and_abstract} ({(title_and_abstract / textxml_present) * 100:.2f}%)"
    #    )
    #    print(
    #        f"More than 'Title' and 'Abstract': {more_than_title_abstract} ({(more_than_title_abstract / textxml_present) * 100:.2f}%)\n"
    #    )

    # License statistics
    print("License Distribution:")
    for license_value, count in license_counter.items():
        license_percentage = (count / total_articles) * 100
        label = license_value if license_value else "None"
        print(f"License: {label} - {license_percentage:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    analyze_json_files(args.input_path)

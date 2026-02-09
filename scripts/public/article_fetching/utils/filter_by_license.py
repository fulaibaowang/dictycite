#!/usr/bin/env python3
import os
import json
import shutil
import argparse
from typing import Set

# Define invalid license list
INVALID_LICENSES: Set[str] = {"cc", "cc by-nd", "cc by-nc-nd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter JSON files based on license and content."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the source directory containing JSON files.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to the destination directory for filtered files.",
    )

    return parser.parse_args()


def filter_json_files(source_dir: str, dest_dir: str) -> None:
    """
    Filters JSON files from source_dir and copies valid ones to dest_dir.

    A file is considered valid if it contains:
    - a 'license' not in INVALID_LICENSES (case-insensitive)

    Args:
        source_dir: Source directory containing JSON files
        dest_dir: Destination directory for filtered files
    """
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if not filename.endswith(".json"):
            continue

        src_file = os.path.join(source_dir, filename)
        try:
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            license = data.get("license")
            if license is not None and license.lower() in INVALID_LICENSES:
                print(f"Invalid license ({license})):", filename)
                continue

            # Check license
            license_value = data.get("license")
            if license_value is not None and license_value.lower() in INVALID_LICENSES:
                print(f"Skipping (invalid license '{license_value}'): {filename}")
                continue

            dest_file = os.path.join(dest_dir, filename)
            shutil.copy2(src_file, dest_file)

        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"Skipping (error): {filename} - {e}")


def main():

    args = parse_args()

    if args.output_path is None:
        args.output_path = args.input_path + "_filtered"

    filter_json_files(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

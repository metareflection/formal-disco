#!/usr/bin/env python3
"""
Push agenda-programs.json to the Hugging Face Hub as a Dataset.

python scripts/dataset_to_hub.py --input results/agenda-programs.json --repo metareflection/dafny-disco
"""

import argparse
import json
import re

from datasets import Dataset


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)

    records = []
    groups = set()

    for item in raw:
        props = item.get("properties", {})
        if props["verification_status"] == "success":
            records.append(
                    {
                        "id": item["path"].split('/', 1)[-1],
                        "group": props.get("parent_idea", ""),
                        "content": item["content"],
                    }
            )
            groups.add(records[-1]["group"])

    print(len(groups), 'ideas/groups')

    return records


def print_stats(records: list[dict]):
    groups = {r["group"] for r in records}
    combined = "\n".join(r["content"] for r in records)
    stats = {
        "programs": len(records),
        "groups": len(groups),
        "classes": len(re.findall(r"^\s*class\s+\w+", combined, re.MULTILINE)),
        "methods": len(re.findall(r"^\s*method\s+\w+", combined, re.MULTILINE)),
        "functions": len(re.findall(r"^\s*function\s+\w+", combined, re.MULTILINE)),
        "lemmas": len(re.findall(r"^\s*lemma\s+\w+", combined, re.MULTILINE)),
        "predicates": len(re.findall(r"^\s*predicate\s+\w+", combined, re.MULTILINE)),
        "asserts": len(re.findall(r"\bassert\b", combined)),
        "invariants": len(re.findall(r"\binvariant\b", combined)),
    }
    print("| Statistic | Count |")
    print("|-----------|------:|")
    for key, val in stats.items():
        print(f"| {key.capitalize()} | {val:,} |")


def main():
    parser = argparse.ArgumentParser(description="Push dataset of Dafny programs to HF Hub")
    parser.add_argument(
        "--input",
        default="results/agenda-programs.json",
        help="Path to the JSON file (default: results/agenda-programs.json)",
    )
    parser.add_argument(
        "--repo",
        default="metareflection/dafny-disco",
        help="HF Hub dataset repo name (default: metareflection/dafny-disco)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics before pushing",
    )
    args = parser.parse_args()

    print(f"Loading records from {args.input} ...")
    records = load_records(args.input)
    print(f"Loaded {len(records):,} records.")

    if args.stats:
        print_stats(records)

    dataset = Dataset.from_list(records)
    print(dataset)

    if input('Push to hugging face? (goforit/no)') == 'goforit':
        dataset.push_to_hub(args.repo)
        print(f"Pushed to https://huggingface.co/datasets/{args.repo}")
    else:
        print('Not goingforit')


if __name__ == "__main__":
    main()

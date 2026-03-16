#!/usr/bin/env python3
"""
Convert lemma_synth examples to repair format for compatibility with distill.py.

Takes lemma_train.pkl / lemma_val.pkl and converts each example from
prompt="lemma_synth" (with raw body as response) to prompt="repair"
(with a diff as response).

Usage:
    python lemma_to_repair.py lemma_train.pkl -o lemma_repair_train.pkl
    python lemma_to_repair.py lemma_val.pkl -o lemma_repair_val.pkl
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from agenda import Object
from patch import apply_text_diff


def build_lemma_diff(hollowed: str, body: str, lemma_name: str) -> str | None:
    """
    Build a diff that inserts the lemma body into the hollowed program.

    Uses the lemma signature lines as anchors to avoid ambiguity.
    """
    lines = hollowed.split('\n')

    # Find the lemma declaration
    lemma_idx = None
    for i, line in enumerate(lines):
        if re.match(rf'\s*lemma\s+{re.escape(lemma_name)}\b', line):
            lemma_idx = i
            break

    if lemma_idx is None:
        return None

    # Find the opening { line
    brace_line = None
    for i in range(lemma_idx, len(lines)):
        if '{' in lines[i]:
            brace_line = i
            break

    if brace_line is None:
        return None

    # Build anchor: use the lemma declaration line and any requires/ensures
    # up to and including the { line. Use enough lines to be unique.
    # Start with the line just before {, or the lemma line itself.
    if brace_line > lemma_idx:
        anchor_line = lines[brace_line - 1]
    else:
        anchor_line = lines[brace_line]

    # Build diff: anchor on a line near the {, then add body lines
    diff_parts = [f"@@{anchor_line}@@"]

    # If anchor wasn't the { line itself, keep the { line
    if brace_line > lemma_idx and lines[brace_line].strip() == '{':
        diff_parts.append(f"= {lines[brace_line]}")

    # Add body lines
    for body_line in body.split('\n'):
        diff_parts.append(f"+ {body_line}")

    return '\n'.join(diff_parts)


def convert_example(ex: dict) -> dict | None:
    """Convert a lemma_synth example to repair format."""
    hollowed = ex['arguments']['program']
    body = ex['response']
    lemma_name = ex['arguments']['lemma_name']
    notes = ex['arguments']['notes']

    if not body.strip():
        return None

    diff = build_lemma_diff(hollowed, body, lemma_name)
    if diff is None:
        return None

    # Verify diff round-trips
    try:
        result = apply_text_diff(hollowed, diff)
        # Check that the body appears in the result
        if body.strip() not in result:
            return None
    except Exception:
        return None

    return {
        "prompt": "repair",
        "arguments": {
            "program": hollowed,
            "notes": notes,
        },
        "response": diff,
        "outcome": "success",
        "metadata": ex.get('metadata', {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert lemma_synth examples to repair format",
    )
    parser.add_argument("input", type=Path, help="Input lemma pkl file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output pkl file")
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    stats = Counter()
    converted = []

    for _, obj in tqdm(data['objects'].items(), desc="Converting"):
        content = obj.content.decode('utf-8') if isinstance(obj.content, bytes) else str(obj.content)
        ex = json.loads(content)

        if ex.get('prompt') != 'lemma_synth':
            stats['skip_wrong_type'] += 1
            continue

        result = convert_example(ex)
        if result is None:
            stats['skip_bad_diff'] += 1
            continue

        converted.append(result)
        stats['converted'] += 1

    # Save
    objects = {}
    for i, ex in enumerate(converted):
        p = f"distil/lemma_repair_{i:06d}"
        c = json.dumps(ex, ensure_ascii=False).encode("utf-8")
        objects[p] = Object(path=p, type="distil-example", content=c)

    pickle_data = {
        "objects": objects,
        "tasks": {},
        "status": {},
        "clock": 0,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(pickle_data, f)

    print(f"\n{'='*50}")
    print("CONVERSION SUMMARY")
    print(f"{'='*50}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

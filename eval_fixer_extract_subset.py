#!/usr/bin/env python3
"""
Extract a subset of program names from eval_fixer.py results.

Usage:
    # Get all successful programs + same number of failures
    python eval_fixer_extract_subset.py baseline.json > subset.txt

    # Get only successful programs
    python eval_fixer_extract_subset.py baseline.json --successes-only > successes.txt

    # Get only failed programs
    python eval_fixer_extract_subset.py baseline.json --failures-only > failures.txt

    # Custom ratio of failures to successes (e.g., 2x as many failures)
    python eval_fixer_extract_subset.py baseline.json --failure-ratio 2.0 > subset.txt
"""

import json
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Extract program names from eval_fixer.py results'
    )
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--successes-only', action='store_true',
                        help='Output only successful programs')
    parser.add_argument('--failures-only', action='store_true',
                        help='Output only failed programs')
    parser.add_argument('--failure-ratio', type=float, default=1.0,
                        help='Ratio of failures to successes (default: 1.0)')
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    successes = [r["program_name"] for r in results if r.get("success")]
    failures = [r["program_name"] for r in results if not r.get("success")]

    if args.successes_only:
        subset = successes
    elif args.failures_only:
        subset = failures
    else:
        # successes + (ratio * len(successes)) failures
        num_failures = int(len(successes) * args.failure_ratio)
        subset = successes + failures[:num_failures]

    for name in subset:
        print(name)

    # Summary to stderr
    print(f"# Extracted {len(subset)} programs: {len(successes)} successes", file=sys.stderr)
    if not args.successes_only and not args.failures_only:
        num_failures_included = min(len(failures), int(len(successes) * args.failure_ratio))
        print(f"# + {num_failures_included} failures (ratio={args.failure_ratio})", file=sys.stderr)


if __name__ == '__main__':
    main()

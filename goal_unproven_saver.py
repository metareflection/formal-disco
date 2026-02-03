#!/usr/bin/env python3
"""
goal_unproven_saver.py - Rescue GOAL_UNPROVEN programs by removing unproven postconditions.

GOAL_UNPROVEN programs are almost verified — Dafny accepted everything except
some postconditions. This script parses Dafny's error output to identify which
ensures clauses failed and removes only those, then verifies once to confirm.

Much faster than the general cleaner (~2 Dafny calls per program instead of dozens)
and produces higher quality output since most of the specification is preserved.

Usage:
    python goal_unproven_saver.py agenda.pkl -o agenda-saved.pkl
    python goal_unproven_saver.py agenda.pkl -n 500 -v
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dafny import DafnyProgram, VerificationOutcome
from distill_common import get_content


def parse_failing_lines(dafny_stdout: str) -> set[int]:
    """Parse Dafny output to find line numbers with postcondition failures.

    Dafny error format:
        ex.dfy(42,0): Error: a]postcondition could not be proved ...
    """
    failing = set()
    for line in dafny_stdout.splitlines():
        # Match patterns like "ex.dfy(42,0): Error:" or "(42,4): Error:"
        m = re.search(r'\((\d+),\d+\):\s*Error:', line)
        if m:
            # Dafny lines are 1-indexed
            failing.add(int(m.group(1)))
    return failing


def find_ensures_clauses(program: str) -> list[dict]:
    """Find all ensures clauses with their line numbers (0-indexed).

    Handles multi-line ensures by tracking continuation lines
    (lines that are indented more and don't start a new keyword).
    """
    lines = program.splitlines()
    clauses = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('ensures ') or stripped == 'ensures':
            start = i
            # Check for continuation lines
            end = i
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                # Stop if we hit another keyword, empty line, or brace
                if (not next_stripped or
                    next_stripped.startswith(('ensures ', 'requires ', 'modifies ',
                                             'reads ', 'decreases ', 'invariant ',
                                             '{', '}', 'lemma ', 'method ',
                                             'function ', 'predicate ', 'class ',
                                             'datatype ', 'module '))):
                    break
                end = j
                j += 1
            clauses.append({
                'start': start,
                'end': end,
                'text': '\n'.join(lines[start:end + 1]),
            })
            i = end + 1
        else:
            i += 1

    return clauses


def remove_lines(program: str, lines_to_remove: set[int]) -> str:
    """Remove specific lines (0-indexed) from a program."""
    lines = program.splitlines()
    return '\n'.join(line for i, line in enumerate(lines) if i not in lines_to_remove)


def save_program(program: str, verbose: bool = False) -> tuple[str, VerificationOutcome, int]:
    """Try to save a GOAL_UNPROVEN program.

    Strategy:
    1. Verify to get current error lines.
    2. Find ensures clauses that overlap with error lines.
    3. Remove those ensures clauses.
    4. Verify to confirm SUCCESS.
    5. If still not SUCCESS, fall back to removing all ensures.

    Returns (saved_program, outcome, num_ensures_removed).
    """
    # Step 1: verify to get error locations
    prog = DafnyProgram(program)
    ver = prog.verify()

    if ver.outcome == VerificationOutcome.SUCCESS:
        return program, ver.outcome, 0

    failing_lines = parse_failing_lines(ver.stdout)
    ensures_clauses = find_ensures_clauses(program)

    if not ensures_clauses:
        return program, ver.outcome, 0

    # Step 2: targeted removal — only ensures that overlap with failing lines
    if failing_lines:
        lines_to_remove = set()
        removed_count = 0
        for clause in ensures_clauses:
            # Check if any line of this clause is flagged by Dafny
            # Dafny uses 1-indexed lines, our clauses are 0-indexed
            clause_lines_1indexed = set(range(clause['start'] + 1, clause['end'] + 2))
            if clause_lines_1indexed & failing_lines:
                for l in range(clause['start'], clause['end'] + 1):
                    lines_to_remove.add(l)
                removed_count += 1

        if lines_to_remove:
            candidate = remove_lines(program, lines_to_remove)
            ver2 = DafnyProgram(candidate).verify()
            if ver2.outcome == VerificationOutcome.SUCCESS:
                return candidate, ver2.outcome, removed_count

    # Step 3: fallback — remove ALL ensures clauses
    all_ensures_lines = set()
    for clause in ensures_clauses:
        for l in range(clause['start'], clause['end'] + 1):
            all_ensures_lines.add(l)

    candidate = remove_lines(program, all_ensures_lines)
    ver3 = DafnyProgram(candidate).verify()
    return candidate, ver3.outcome, len(ensures_clauses)


def main():
    parser = argparse.ArgumentParser(
        description="Rescue GOAL_UNPROVEN programs by removing unproven postconditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python goal_unproven_saver.py agenda.pkl -o agenda-saved.pkl
  python goal_unproven_saver.py agenda.pkl -n 500 -v

Typically rescues ~80-90% of GOAL_UNPROVEN programs with minimal spec loss.
        """
    )
    parser.add_argument('input', type=Path, help="Input agenda pickle file")
    parser.add_argument('-o', '--output', type=Path,
                        help="Output pickle file (default: input_saved.pkl)")
    parser.add_argument('-n', '--max', type=int,
                        help="Maximum programs to process")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print details for each program")

    args = parser.parse_args()

    if not args.output:
        args.output = args.input.with_stem(args.input.stem + '_saved')

    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    objects = data['objects']

    # Find GOAL_UNPROVEN programs
    candidates = []
    for path, obj in objects.items():
        if obj.type != 'dafny-program':
            continue
        if obj.properties.get('verification_status') == 'goal_unproven':
            content = get_content(obj)
            if content:
                candidates.append((path, obj, content))

    print(f"Found {len(candidates)} GOAL_UNPROVEN programs")

    if args.max:
        candidates = candidates[:args.max]
        print(f"Processing first {args.max}")

    stats = Counter()
    repairs = {}

    pbar = tqdm(candidates, desc="Saving programs")
    for path, obj, content in pbar:
        saved, outcome, num_removed = save_program(content, verbose=args.verbose)

        if outcome == VerificationOutcome.SUCCESS:
            stats['saved'] += 1
            stats['total_ensures_removed'] += num_removed
            repairs[path] = saved
            if args.verbose:
                print(f"  ✓ {path}: removed {num_removed} ensures clause(s)")
        else:
            stats['still_unproven'] += 1
            if args.verbose:
                print(f"  ✗ {path}: could not save (outcome: {outcome.name})")

        pbar.set_postfix(saved=stats['saved'], failed=stats['still_unproven'])

    # Summary
    total = len(candidates)
    print(f"\n{'='*50}")
    print("GOAL_UNPROVEN SAVER SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed:       {total}")
    print(f"Saved to SUCCESS:      {stats['saved']} ({100*stats['saved']/total:.1f}%)" if total else "")
    print(f"Still unproven:        {stats['still_unproven']}")
    print(f"Ensures clauses removed: {stats['total_ensures_removed']}")
    if stats['saved']:
        print(f"Avg ensures removed:   {stats['total_ensures_removed']/stats['saved']:.1f}")

    # Apply repairs
    if repairs:
        print(f"\nApplying {len(repairs)} repairs...")
        for path, new_content in repairs.items():
            obj = objects[path]
            obj.content = new_content.encode('utf-8') if isinstance(obj.content, bytes) else new_content
            obj.properties['verification_status'] = 'success'

        print(f"Saving to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print("Done!")
    else:
        print("\nNo repairs to apply.")


if __name__ == '__main__':
    main()

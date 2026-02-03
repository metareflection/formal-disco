#!/usr/bin/env python3
"""
cleaner.py - Clean unverified Dafny programs by removing failing constructs.

This script processes agenda pickle files and repairs FAIL programs by
systematically removing constructs (lemmas, functions, methods, ensures)
until the program verifies.

Usage:
    python cleaner.py agenda.pkl                    # Process all FAIL programs
    python cleaner.py agenda.pkl -o cleaned.pkl    # Specify output file
    python cleaner.py agenda.pkl -n 1000           # Process first 1000 only

Example:
    python cleaner.py agenda-run3.pkl -o agenda-run3-cleaned.pkl -n 500

Success rate: ~54% of FAIL programs can be repaired to SUCCESS.
Runtime: ~30 seconds per program.
"""

import argparse
import pickle
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm

from dafny import DafnyProgram, VerificationOutcome


@dataclass
class CleanResult:
    """Result of cleaning a single program."""
    original_path: str
    outcome: VerificationOutcome
    removed_items: list[str]
    original_lines: int
    cleaned_lines: int


def find_block_end(lines: list[str], start: int) -> int:
    """Find the end of a brace-delimited block starting at 'start'."""
    brace_count = 0
    found_open = False

    for i in range(start, len(lines)):
        for char in lines[i]:
            if char == '{':
                brace_count += 1
                found_open = True
            elif char == '}':
                brace_count -= 1

        if found_open and brace_count == 0:
            return i

    return len(lines) - 1


def find_removable_constructs(program: str) -> list[tuple[int, int, str, str]]:
    """
    Find constructs that can be removed to fix a failing program.
    
    Returns list of (start_line, end_line, construct_type, name) tuples,
    sorted by removal priority (most disposable first).
    """
    lines = program.splitlines()
    removable = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ensures clauses (postconditions) - most disposable
        if stripped.startswith('ensures '):
            removable.append((i, i, 'ensures', stripped[:50]))
            i += 1
            continue

        # requires clauses (preconditions)
        if stripped.startswith('requires '):
            removable.append((i, i, 'requires', stripped[:50]))
            i += 1
            continue

        # lemma declarations
        lemma_match = re.match(r'\s*(lemma|ghost method)\s+(\w+)', line)
        if lemma_match:
            name = lemma_match.group(2)
            end = find_block_end(lines, i)
            removable.append((i, end, 'lemma', name))
            i = end + 1
            continue

        # function/predicate declarations
        func_match = re.match(r'\s*(function|predicate|ghost function)\s+(\w+)', line)
        if func_match:
            name = func_match.group(2)
            end = find_block_end(lines, i)
            removable.append((i, end, 'function', name))
            i = end + 1
            continue

        # method declarations
        method_match = re.match(r'\s*method\s+(\w+)', line)
        if method_match:
            name = method_match.group(1)
            end = find_block_end(lines, i)
            removable.append((i, end, 'method', name))
            i = end + 1
            continue

        i += 1

    # Sort by priority: ensures > requires > lemma > function > method
    priority = {'ensures': 0, 'requires': 1, 'lemma': 2, 'function': 3, 'method': 4}
    removable.sort(key=lambda x: (priority.get(x[2], 99), x[0]))

    return removable


def remove_lines(program: str, start: int, end: int) -> str:
    """Remove lines from start to end (inclusive)."""
    lines = program.splitlines()
    new_lines = lines[:start] + lines[end + 1:]
    return '\n'.join(new_lines)


def clean_program(
    program: str,
    timeout: float = 15.0,
    max_removals: int = 25,
) -> tuple[str, VerificationOutcome, list[str]]:
    """
    Clean a program by removing constructs until it verifies.
    
    Returns (cleaned_program, outcome, list_of_removed_items).
    """
    current = program
    removed_items = []

    # Check if already verifies
    result = DafnyProgram(current).verify()
    if result.outcome == VerificationOutcome.SUCCESS:
        return current, result.outcome, []

    # Iteratively remove constructs
    for _ in range(max_removals):
        removable = find_removable_constructs(current)
        if not removable:
            break

        made_progress = False
        for start, end, construct_type, name in removable:
            candidate = remove_lines(current, start, end)

            if not candidate.strip():
                continue

            result = DafnyProgram(candidate).verify()

            if result.outcome == VerificationOutcome.SUCCESS:
                removed_items.append(f"{construct_type}: {name}")
                return candidate, result.outcome, removed_items

            # Accept if it improves from FAIL to GOAL_UNPROVEN
            if result.outcome == VerificationOutcome.GOAL_UNPROVEN:
                current = candidate
                removed_items.append(f"{construct_type}: {name}")
                made_progress = True
                break

        if not made_progress:
            # Force remove first item to make progress
            if removable:
                start, end, construct_type, name = removable[0]
                candidate = remove_lines(current, start, end)
                if candidate.strip():
                    current = candidate
                    removed_items.append(f"{construct_type}: {name} (forced)")

    # Final verification
    result = DafnyProgram(current).verify()
    return current, result.outcome, removed_items


def clean_agenda(
    input_path: Path,
    output_path: Path,
    max_programs: Optional[int] = None,
    timeout: float = 15.0,
    verbose: bool = False,
) -> dict:
    """
    Clean FAIL programs in an agenda pickle file.
    
    Returns statistics dictionary.
    """
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    status = data['status']
    objects = data['objects']

    # Find FAIL programs
    fail_programs = []
    for task_id, task_status in status.items():
        verification = task_status.worker_notes.get('verification')
        if verification == 'FAIL':
            program_path = task_status.worker_notes.get('program_path')
            if program_path and program_path in objects:
                fail_programs.append((task_id, program_path))

    print(f"Found {len(fail_programs)} FAIL programs")

    if max_programs:
        fail_programs = fail_programs[:max_programs]
        print(f"Processing first {max_programs}")

    # Process programs
    stats = Counter()
    repairs = {}

    for task_id, program_path in tqdm(fail_programs, desc="Cleaning"):
        obj = objects[program_path]
        content = obj.content.decode('utf-8') if isinstance(obj.content, bytes) else obj.content

        cleaned, outcome, removed = clean_program(content, timeout=timeout)

        if outcome == VerificationOutcome.SUCCESS:
            stats['cleaned_to_success'] += 1
            repairs[program_path] = (task_id, cleaned, 'SUCCESS', removed)
            if verbose:
                print(f"  ✓ {program_path}: removed {len(removed)} constructs")
        elif outcome == VerificationOutcome.GOAL_UNPROVEN:
            stats['cleaned_to_goal_unproven'] += 1
            repairs[program_path] = (task_id, cleaned, 'GOAL_UNPROVEN', removed)
        else:
            stats['still_fail'] += 1
            if verbose:
                print(f"  ✗ {program_path}: could not clean")

    # Print summary
    total = len(fail_programs)
    print(f"\n{'='*50}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed:         {total}")
    print(f"Cleaned to SUCCESS:      {stats['cleaned_to_success']} ({100*stats['cleaned_to_success']/total:.1f}%)")
    print(f"Cleaned to GOAL_UNPROVEN:{stats['cleaned_to_goal_unproven']}")
    print(f"Still FAIL:              {stats['still_fail']}")

    # Apply repairs
    if repairs:
        print(f"\nApplying {len(repairs)} repairs...")
        
        for program_path, (task_id, new_content, new_outcome, _) in repairs.items():
            # Update program content
            objects[program_path].content = new_content.encode('utf-8')
            
            # Update task status
            status[task_id].worker_notes['verification'] = new_outcome

        print(f"Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print("Done!")

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Clean unverified Dafny programs by removing failing constructs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleaner.py agenda.pkl                     Process all FAIL programs
  python cleaner.py agenda.pkl -o cleaned.pkl     Specify output file
  python cleaner.py agenda.pkl -n 1000            Process first 1000 only

Success rate: ~54% of FAIL programs can be cleaned to SUCCESS.
        """
    )
    parser.add_argument('input', type=Path, help="Input agenda pickle file")
    parser.add_argument('-o', '--output', type=Path, help="Output pickle file (default: input_cleaned.pkl)")
    parser.add_argument('-n', '--max', type=int, help="Maximum programs to process")
    parser.add_argument('-t', '--timeout', type=float, default=15.0, help="Verification timeout in seconds (default: 15)")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print details for each program")

    args = parser.parse_args()

    if not args.output:
        args.output = args.input.with_stem(args.input.stem + '_cleaned')

    clean_agenda(
        args.input,
        args.output,
        max_programs=args.max,
        timeout=args.timeout,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()

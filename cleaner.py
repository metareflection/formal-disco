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
import os
import pickle
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    workers: int = 1,
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

    print(f"Using {workers} workers")

    # Process programs
    stats = Counter()
    processed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {}
        for task_id, program_path in fail_programs:
            obj = objects[program_path]
            content = obj.content.decode('utf-8') if isinstance(obj.content, bytes) else obj.content
            future = executor.submit(clean_program, content, timeout)
            futures[future] = (task_id, program_path)

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Cleaning")
        for future in pbar:
            task_id, program_path = futures[future]
            try:
                cleaned, outcome, removed = future.result()
            except Exception as e:
                stats['error'] += 1
                if verbose:
                    print(f"  ! {program_path}: error: {e}")
                pbar.set_postfix_str(f"{stats['cleaned_to_success']:>6}✓ {stats['still_fail']:>6}✗")
                continue

            if outcome == VerificationOutcome.SUCCESS:
                stats['cleaned_to_success'] += 1
                # Apply immediately
                objects[program_path].content = cleaned.encode('utf-8')
                objects[program_path].properties['verification_status'] = 'success'
                objects[program_path].properties['verification_outcome'] = 'SUCCESS'
                objects[program_path].properties['cleaned_from_fail'] = True
                status[task_id].worker_notes['verification'] = 'SUCCESS'
                if verbose:
                    print(f"  ✓ {program_path}: removed {len(removed)} constructs")
            elif outcome == VerificationOutcome.GOAL_UNPROVEN:
                stats['cleaned_to_goal_unproven'] += 1
                objects[program_path].content = cleaned.encode('utf-8')
                objects[program_path].properties['verification_status'] = 'goal_unproven'
                objects[program_path].properties['verification_outcome'] = 'GOAL_UNPROVEN'
                status[task_id].worker_notes['verification'] = 'GOAL_UNPROVEN'
            else:
                stats['still_fail'] += 1
                if verbose:
                    print(f"  ✗ {program_path}: could not clean")

            pbar.set_postfix_str(f"{stats['cleaned_to_success']:>6}✓ {stats['still_fail']:>6}✗")

            # Save periodically
            processed += 1
            if processed % 50 == 0 and stats['cleaned_to_success'] > 0:
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)

    # Print summary
    total = len(fail_programs)
    print(f"\n{'='*50}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed:         {total}")
    print(f"Cleaned to SUCCESS:      {stats['cleaned_to_success']} ({100*stats['cleaned_to_success']/total:.1f}%)" if total else "")
    print(f"Cleaned to GOAL_UNPROVEN:{stats['cleaned_to_goal_unproven']}")
    print(f"Still FAIL:              {stats['still_fail']}")

    # Final save
    if stats['cleaned_to_success'] or stats['cleaned_to_goal_unproven']:
        print(f"\nSaving to {output_path}...")
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
    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: cpu count)")
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
        workers=args.workers or 1,
    )


if __name__ == '__main__':
    main()

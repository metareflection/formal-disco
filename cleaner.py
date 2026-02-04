#!/usr/bin/env python3
"""
cleaner.py - Clean unverified Dafny programs by removing failing constructs.

This script processes agenda pickle files and repairs FAIL programs by
using Dafny's error output to identify and remove failing constructs.

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


@dataclass
class DafnyError:
    """A parsed error from Dafny's output."""
    line: int
    column: int
    error_type: str
    message: str


def parse_dafny_errors(stdout: str) -> list[DafnyError]:
    """
    Parse Dafny's stdout to extract error locations and types.

    Dafny errors typically look like:
        program.dfy(15,4): Error: assertion might not hold
        program.dfy(23,8): Error BP5003: postcondition might not hold
    """
    errors = []
    # Match patterns like: filename.dfy(line,col): Error...
    pattern = r'\.dfy\((\d+),(\d+)\):\s*(Error|Warning)[^:]*:\s*(.+)'

    for match in re.finditer(pattern, stdout):
        line = int(match.group(1)) - 1  # Convert to 0-indexed
        column = int(match.group(2))
        error_type = match.group(3)
        message = match.group(4).strip()
        errors.append(DafnyError(line, column, error_type, message))

    return errors


def count_ensures_clauses(program: str) -> int:
    """Count the number of 'ensures' clauses in a program."""
    count = 0
    for line in program.splitlines():
        if line.strip().startswith('ensures '):
            count += 1
    return count


def has_ensures_clause(program: str) -> bool:
    """Check if a program has at least one 'ensures' clause."""
    return count_ensures_clauses(program) > 0


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


@dataclass
class Construct:
    """A removable construct in a Dafny program."""
    start_line: int
    end_line: int
    construct_type: str
    name: str

    def contains_line(self, line: int) -> bool:
        """Check if this construct contains the given line."""
        return self.start_line <= line <= self.end_line


def find_all_constructs(program: str) -> list[Construct]:
    """
    Find all constructs in a program that could potentially be removed.

    Returns list of Construct objects.
    """
    lines = program.splitlines()
    constructs = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ensures clauses (postconditions)
        if stripped.startswith('ensures '):
            constructs.append(Construct(i, i, 'ensures', stripped[:50]))
            i += 1
            continue

        # requires clauses (preconditions)
        if stripped.startswith('requires '):
            constructs.append(Construct(i, i, 'requires', stripped[:50]))
            i += 1
            continue

        # assert statements
        if stripped.startswith('assert '):
            constructs.append(Construct(i, i, 'assert', stripped[:50]))
            i += 1
            continue

        # invariant clauses
        if stripped.startswith('invariant '):
            constructs.append(Construct(i, i, 'invariant', stripped[:50]))
            i += 1
            continue

        # lemma declarations
        lemma_match = re.match(r'\s*(lemma|ghost method)\s+(\w+)', line)
        if lemma_match:
            name = lemma_match.group(2)
            end = find_block_end(lines, i)
            constructs.append(Construct(i, end, 'lemma', name))
            i = end + 1
            continue

        # function/predicate declarations
        func_match = re.match(r'\s*(function|predicate|ghost function)\s+(\w+)', line)
        if func_match:
            name = func_match.group(2)
            end = find_block_end(lines, i)
            constructs.append(Construct(i, end, 'function', name))
            i = end + 1
            continue

        # method declarations
        method_match = re.match(r'\s*method\s+(\w+)', line)
        if method_match:
            name = method_match.group(1)
            end = find_block_end(lines, i)
            constructs.append(Construct(i, end, 'method', name))
            i = end + 1
            continue

        i += 1

    return constructs


def find_constructs_with_errors(
    constructs: list[Construct],
    errors: list[DafnyError],
) -> list[Construct]:
    """
    Find constructs that contain errors, sorted by priority.

    Priority order for removal:
    1. assert statements with errors
    2. invariants with errors
    3. ensures clauses with errors
    4. requires clauses with errors
    5. lemmas with errors
    6. functions with errors
    7. methods with errors
    """
    priority = {
        'assert': 0,
        'invariant': 1,
        'ensures': 2,
        'requires': 3,
        'lemma': 4,
        'function': 5,
        'method': 6,
    }

    error_lines = {e.line for e in errors}

    # Find constructs containing error lines
    constructs_with_errors = []
    for construct in constructs:
        for error_line in error_lines:
            if construct.contains_line(error_line):
                constructs_with_errors.append(construct)
                break

    # Sort by priority
    constructs_with_errors.sort(key=lambda c: (priority.get(c.construct_type, 99), c.start_line))

    return constructs_with_errors


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
    Clean a program by using Dafny's error output to identify and remove
    failing constructs.

    Strategy:
    1. Run Dafny and parse error locations from stdout
    2. Find constructs containing those error lines
    3. Remove the highest-priority failing construct (preserving at least one ensures)
    4. Repeat until success or no progress

    Programs must retain at least one 'ensures' clause to be considered valid.

    Returns (cleaned_program, outcome, list_of_removed_items).
    """
    current = program
    removed_items = []

    # Check if program has any ensures clauses - reject if none
    if not has_ensures_clause(current):
        return current, VerificationOutcome.FAIL, []

    # Check if already verifies
    result = DafnyProgram(current).verify()
    if result.outcome == VerificationOutcome.SUCCESS:
        return current, result.outcome, []

    # Iteratively remove failing constructs
    for _ in range(max_removals):
        # Parse errors from Dafny output
        errors = parse_dafny_errors(result.stdout)
        all_constructs = find_all_constructs(current)

        if not all_constructs:
            break

        # Count ensures clauses to know if we can remove any
        ensures_count = count_ensures_clauses(current)

        # Find constructs containing errors
        if errors:
            failing_constructs = find_constructs_with_errors(all_constructs, errors)
        else:
            # No parseable errors - fall back to priority order
            failing_constructs = []

        # If no specific failing constructs found, use all constructs in priority order
        if not failing_constructs:
            priority = {
                'assert': 0,
                'invariant': 1,
                'ensures': 2,
                'requires': 3,
                'lemma': 4,
                'function': 5,
                'method': 6,
            }
            failing_constructs = sorted(
                all_constructs,
                key=lambda c: (priority.get(c.construct_type, 99), c.start_line)
            )

        made_progress = False
        for construct in failing_constructs:
            # Don't remove the last ensures clause
            if construct.construct_type == 'ensures' and ensures_count <= 1:
                continue

            candidate = remove_lines(current, construct.start_line, construct.end_line)

            if not candidate.strip():
                continue

            # Verify candidate still has ensures clause
            if not has_ensures_clause(candidate):
                continue

            result = DafnyProgram(candidate).verify()

            if result.outcome == VerificationOutcome.SUCCESS:
                removed_items.append(f"{construct.construct_type}: {construct.name}")
                return candidate, result.outcome, removed_items

            # Accept if it reduces errors or improves outcome
            new_errors = parse_dafny_errors(result.stdout)
            if (result.outcome == VerificationOutcome.GOAL_UNPROVEN or
                    len(new_errors) < len(errors)):
                current = candidate
                removed_items.append(f"{construct.construct_type}: {construct.name}")
                made_progress = True
                break

        if not made_progress:
            # Force remove first failing construct to make progress
            # (but still respect the ensures constraint)
            for construct in failing_constructs:
                if construct.construct_type == 'ensures' and ensures_count <= 1:
                    continue
                candidate = remove_lines(current, construct.start_line, construct.end_line)
                if candidate.strip() and has_ensures_clause(candidate):
                    current = candidate
                    removed_items.append(f"{construct.construct_type}: {construct.name} (forced)")
                    result = DafnyProgram(current).verify()
                    break

    # Final verification - only accept if program still has ensures
    result = DafnyProgram(current).verify()
    if not has_ensures_clause(current):
        return current, VerificationOutcome.FAIL, removed_items
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

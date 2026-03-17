#!/usr/bin/env python3
"""
Generate distillation data from verified Dafny programs by mutation.

Takes verified programs and creates repair examples by removing invariants/assertions
one at a time, capturing the error and the diff to fix it.

Usage:
    python free_autogen_distill.py generate --output free_autogen.pkl
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from agenda import Object
from execute import get_dafny_errors


def find_removable_lines(program: str) -> list[tuple[int, str, str]]:
    """
    Find lines that can be removed to create broken programs.

    Returns list of (line_index, line_content, line_type) where line_type is
    'invariant', 'assert', 'decreases', etc.
    """
    lines = program.splitlines()
    removable = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Match invariant lines
        if stripped.startswith('invariant '):
            removable.append((i, line, 'invariant'))
        # Match assert statements
        elif stripped.startswith('assert '):
            removable.append((i, line, 'assert'))
        # Match decreases clauses
        elif stripped.startswith('decreases '):
            removable.append((i, line, 'decreases'))

    return removable


def remove_line(program: str, line_idx: int) -> str:
    """Remove a specific line from the program."""
    lines = program.splitlines()
    del lines[line_idx]
    return '\n'.join(lines)


def compute_insertion_diff(program: str, line_idx: int, line_content: str) -> str:
    """
    Compute a diff that inserts the removed line back.

    Uses the line before as anchor, then adds the removed line.
    """
    lines = program.splitlines()

    # Find anchor - the line before where we insert
    if line_idx > 0:
        anchor = lines[line_idx - 1]
    else:
        return f"@@@@\n+ {line_content}"

    return f"@@{anchor}@@\n+ {line_content}"


def generate_mutation_examples(
    input_dir: Path,
    skip_dafny: bool = False,
    max_files: Optional[int] = None,
) -> list[dict]:
    """Generate repair examples by mutating verified programs."""

    dfy_files = sorted(input_dir.glob("*.dfy"))
    if max_files:
        dfy_files = dfy_files[:max_files]

    examples = []
    skipped_no_removable = 0
    skipped_still_verifies = 0

    for dfy_file in tqdm(dfy_files, desc="Processing files"):
        program = dfy_file.read_text()
        removable = find_removable_lines(program)

        if not removable:
            skipped_no_removable += 1
            continue

        for line_idx, line_content, line_type in removable:
            # Create broken version
            broken = remove_line(program, line_idx)

            # Get errors (or check if it still verifies)
            if skip_dafny:
                errors = "(Dafny errors would appear here)"
            else:
                errors = get_dafny_errors(broken, timeout=2)
                # Skip if it still verifies (the line wasn't needed)
                if "0 errors" in errors:
                    skipped_still_verifies += 1
                    continue

            # Compute diff to fix
            # Note: line_idx in broken program is where we need to insert
            diff = compute_insertion_diff(broken, line_idx, line_content)

            examples.append({
                "prompt": "repair",
                "arguments": {
                    "program": broken,
                    "notes": errors,
                },
                "response": diff,
                "outcome": "success",
                "metadata": {
                    "source": "free_autogen",
                    "file": dfy_file.name,
                    "removed_line": line_content.strip(),
                    "line_type": line_type,
                }
            })

    if skipped_no_removable:
        print(f"Skipped {skipped_no_removable} files (no removable lines)")
    if skipped_still_verifies:
        print(f"Skipped {skipped_still_verifies} mutations (still verifies without the line)")

    return examples


def create_agenda_pickle(examples: list[dict], output_path: Path) -> None:
    """Create a pickle file in the format expected by distill.py."""
    objects = {}

    for i, ex in enumerate(examples):
        path = f"distil/repair_{i:04d}"
        content = json.dumps(ex, ensure_ascii=False).encode("utf-8")
        obj = Object(
            path=path,
            type="distil-example",
            content=content,
        )
        objects[path] = obj

    pickle_data = {
        "objects": objects,
        "tasks": {},
        "task_statuses": {},
    }

    with output_path.open("wb") as f:
        pickle.dump(pickle_data, f)


def cmd_generate(args):
    """Generate mutation-based distillation data."""
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    print(f"Generating examples from {input_dir}")

    examples = generate_mutation_examples(
        input_dir,
        skip_dafny=args.skip_dafny,
        max_files=args.max_files,
    )

    print(f"Generated {len(examples)} repair examples")

    create_agenda_pickle(examples, output_path)
    print(f"Saved to {output_path}")

    if examples:
        print("\n--- Sample example ---")
        sample = examples[0]
        print(f"File: {sample['metadata']['file']}")
        print(f"Removed: {sample['metadata']['removed_line']}")
        print(f"Type: {sample['metadata']['line_type']}")
        print(f"\nDiff:\n{sample['response']}")


def cmd_list(args):
    """List removable lines in files."""
    input_dir = Path(args.input_dir)

    for dfy_file in sorted(input_dir.glob("*.dfy"))[:args.max_files or 10]:
        program = dfy_file.read_text()
        removable = find_removable_lines(program)
        print(f"{dfy_file.name}: {len(removable)} removable lines")
        for idx, content, ltype in removable[:3]:
            print(f"  [{ltype}] {content.strip()[:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data by mutation")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="Generate mutation-based distillation data")
    gen_p.add_argument("--input-dir", type=str, default="../dafny-free-autogen",
                       help="Directory containing verified .dfy files")
    gen_p.add_argument("--output", "-o", type=str, default="free_autogen.pkl",
                       help="Output pickle file path")
    gen_p.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of files to process")
    gen_p.add_argument("--skip-dafny", action="store_true",
                       help="Skip running Dafny (faster, but no real error messages)")
    gen_p.set_defaults(func=cmd_generate)

    list_p = sub.add_parser("list", help="List removable lines in files")
    list_p.add_argument("--input-dir", type=str, default="../dafny-free-autogen",
                        help="Directory containing .dfy files")
    list_p.add_argument("--max-files", type=int, default=10,
                        help="Maximum files to show")
    list_p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

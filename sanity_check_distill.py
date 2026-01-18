#!/usr/bin/env python3
"""
Sanity check for distillation pipeline.

Generates synthetic distillation data from DafnyBench (hints_removed -> ground_truth),
then allows training and evaluation on the same data.

Usage:
    # Step 1: Generate synthetic distillation pickle
    python sanity_check_distill.py generate --output sanity_check.pkl

    # Step 2: Check stats
    python distill.py stats -d sanity_check.pkl

    # Step 3: Train (modify config/distill.yaml to point to sanity_check.pkl)
    python distill.py sft data=sanity_check.pkl output_dir=sanity-sft-out

    # Step 4: Eval with the trained model via vLLM
    python eval_fixer.py --llm-config vllm --num-programs 50
"""

import argparse
import difflib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from agenda import Object  # Use the real Object class for pickle compatibility


def compute_text_diff(before: str, after: str) -> str:
    """
    Compute a text diff in the format expected by the repair prompt.

    Format (from patch.py apply_text_diff):
    - @@content@@ anchor (search-forward marker, MUST start and end with @@)
    - = line: keep line (find forward, advance cursor)
    - - line: delete line (find forward, delete)
    - + line: add line (insert at cursor)
    """
    before_lines = before.splitlines(keepends=False)
    after_lines = after.splitlines(keepends=False)

    # Use difflib to get opcodes
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)

    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Skip equal blocks - they don't need to be in the diff
            continue

        # Add anchor with the line just before the change (if exists)
        if i1 > 0:
            anchor_line = before_lines[i1 - 1]
            diff_parts.append(f"@@{anchor_line}@@")
        else:
            # Change at very start of file - use empty anchor
            diff_parts.append("@@@@")

        if tag == 'replace':
            # Delete old lines, add new lines
            for line in before_lines[i1:i2]:
                diff_parts.append(f"- {line}")
            for line in after_lines[j1:j2]:
                diff_parts.append(f"+ {line}")
        elif tag == 'delete':
            for line in before_lines[i1:i2]:
                diff_parts.append(f"- {line}")
        elif tag == 'insert':
            for line in after_lines[j1:j2]:
                diff_parts.append(f"+ {line}")

    return "\n".join(diff_parts)


def get_dafny_errors(program: str, timeout: int = 30) -> str:
    """Run Dafny and capture verification errors."""
    try:
        result = subprocess.run(
            ["dafny", "verify", "/dev/stdin"],
            input=program,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Dafny verification timed out"
    except Exception as e:
        return f"Error running Dafny: {e}"


def generate_repair_examples(
    benchmark_path: Path,
    max_programs: Optional[int] = None,
    skip_dafny: bool = False,
    skip_trivial: bool = False,
    cache_path: Optional[Path] = None,
) -> list[dict]:
    """Generate repair examples from DafnyBench."""
    from patch import apply_text_diff

    gt_dir = benchmark_path / "DafnyBench" / "dataset" / "ground_truth"
    hr_dir = benchmark_path / "DafnyBench" / "dataset" / "hints_removed"

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"Hints removed directory not found: {hr_dir}")

    # Load verification cache if filtering trivial programs
    cache = {}
    if skip_trivial and cache_path and cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)

    examples = []
    skipped_bad_diff = 0
    skipped_trivial = 0
    gt_files = sorted(gt_dir.glob("*.dfy"))

    if max_programs:
        gt_files = gt_files[:max_programs]

    for gt_file in tqdm(gt_files, desc="Generating examples"):
        base = gt_file.stem

        # Find matching hints_removed file (has _no_hints suffix)
        hr_file = hr_dir / f"{base}_no_hints.dfy"
        if not hr_file.exists():
            # Try without _no_hints suffix pattern
            continue

        # Skip trivial programs (already verify) if requested
        hr_name = hr_file.stem
        if skip_trivial and cache.get(hr_name) == "SUCCESS":
            skipped_trivial += 1
            continue

        gt_code = gt_file.read_text()
        hr_code = hr_file.read_text()

        # Skip if they're identical (no hints were removed)
        if gt_code.strip() == hr_code.strip():
            continue

        # Compute diff (what the model should output)
        diff = compute_text_diff(hr_code, gt_code)

        if not diff.strip():
            continue

        # Verify the diff actually produces the correct result
        try:
            result = apply_text_diff(hr_code, diff)
            if result.strip() != gt_code.strip():
                skipped_bad_diff += 1
                continue
        except Exception:
            skipped_bad_diff += 1
            continue

        # Get verification errors for hints_removed version
        if skip_dafny:
            errors = "(Dafny errors would appear here)"
        else:
            errors = get_dafny_errors(hr_code)

        examples.append({
            "prompt": "repair",
            "arguments": {
                "program": hr_code,
                "notes": errors,
            },
            "response": diff,
            "outcome": "success",
            "metadata": {
                "source": "dafnybench",
                "ground_truth_file": str(gt_file),
                "hints_removed_file": str(hr_file),
            }
        })

    if skipped_bad_diff:
        print(f"Skipped {skipped_bad_diff} examples with bad diffs")
    if skipped_trivial:
        print(f"Skipped {skipped_trivial} trivial programs (already verify)")

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
    """Generate synthetic distillation data."""
    benchmark_path = Path(args.benchmark_path)
    output_path = Path(args.output)
    cache_path = Path(args.cache_path) if args.skip_trivial else None

    print(f"Generating examples from {benchmark_path}")
    examples = generate_repair_examples(
        benchmark_path,
        max_programs=args.max_programs,
        skip_dafny=args.skip_dafny,
        skip_trivial=args.skip_trivial,
        cache_path=cache_path,
    )

    print(f"Generated {len(examples)} repair examples")

    create_agenda_pickle(examples, output_path)
    print(f"Saved to {output_path}")

    # Print a sample
    if examples:
        print("\n--- Sample example ---")
        sample = examples[0]
        print(f"Program (first 500 chars):\n{sample['arguments']['program'][:500]}...")
        print(f"\nDiff:\n{sample['response'][:500]}...")


def cmd_verify_diff(args):
    """Verify that generated diffs actually work."""
    from patch import apply_text_diff

    benchmark_path = Path(args.benchmark_path)
    gt_dir = benchmark_path / "DafnyBench" / "dataset" / "ground_truth"
    hr_dir = benchmark_path / "DafnyBench" / "dataset" / "hints_removed"

    gt_files = sorted(gt_dir.glob("*.dfy"))[:args.max_programs or 10]

    success = 0
    fail = 0

    for gt_file in gt_files:
        base = gt_file.stem
        hr_file = hr_dir / f"{base}_no_hints.dfy"
        if not hr_file.exists():
            continue

        gt_code = gt_file.read_text()
        hr_code = hr_file.read_text()

        if gt_code.strip() == hr_code.strip():
            continue

        diff = compute_text_diff(hr_code, gt_code)

        try:
            result = apply_text_diff(hr_code, diff)
            # Check if result matches ground truth
            if result.strip() == gt_code.strip():
                success += 1
                print(f"OK: {base}")
            else:
                fail += 1
                print(f"MISMATCH: {base}")
                if args.verbose:
                    print(f"  Expected:\n{gt_code[:200]}...")
                    print(f"  Got:\n{result[:200]}...")
        except Exception as e:
            fail += 1
            print(f"ERROR: {base}: {e}")
            if args.verbose:
                print(f"  Diff:\n{diff[:300]}...")

    print(f"\nResults: {success} OK, {fail} failed")


def main():
    parser = argparse.ArgumentParser(description="Sanity check for distillation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_p = sub.add_parser("generate", help="Generate synthetic distillation data")
    gen_p.add_argument("--benchmark-path", type=str, default="DafnyBench",
                       help="Path to DafnyBench directory")
    gen_p.add_argument("--output", "-o", type=str, default="sanity_check.pkl",
                       help="Output pickle file path")
    gen_p.add_argument("--max-programs", type=int, default=None,
                       help="Maximum number of programs to process")
    gen_p.add_argument("--skip-dafny", action="store_true",
                       help="Skip running Dafny (faster, but no real error messages)")
    gen_p.add_argument("--skip-trivial", action="store_true",
                       help="Skip programs that already verify (uses cache file)")
    gen_p.add_argument("--cache-path", type=str, default=".fixer_outcome_cache.json",
                       help="Path to verification outcome cache (for --skip-trivial)")
    gen_p.set_defaults(func=cmd_generate)

    # Verify command
    verify_p = sub.add_parser("verify", help="Verify that generated diffs work")
    verify_p.add_argument("--benchmark-path", type=str, default="DafnyBench",
                          help="Path to DafnyBench directory")
    verify_p.add_argument("--max-programs", type=int, default=10,
                          help="Maximum number of programs to verify")
    verify_p.add_argument("--verbose", "-v", action="store_true",
                          help="Show detailed output for failures")
    verify_p.set_defaults(func=cmd_verify_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

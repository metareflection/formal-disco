#!/usr/bin/env python3
"""
VFP distillation data generation.

Generates distillation data from VFP (Verified Function Programming) benchmark
(emptied programs -> ground_truth proofs), similar to sanity_check_distill.py
but for the VFP autogen dataset.

Usage:
    # Step 1: Generate distillation pickle from VFP emptied JSON
    python vfp_distill.py generate --output vfp_distill.pkl

    # Step 2: Check stats
    python distill.py stats -d vfp_distill.pkl

    # Step 3: Train (modify config/distill.yaml to point to vfp_distill.pkl)
    python distill.py sft data=vfp_distill.pkl output_dir=vfp-sft-out
"""

import argparse
import json
import pickle
import re
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from agenda import Object  # Use the real Object class for pickle compatibility


def compute_insertion_diff(program: str, method_name: str, body: str) -> Optional[str]:
    """
    Compute a diff that inserts the body into the empty method/lemma.

    This directly generates the diff by finding the method declaration
    and inserting the body after the opening brace.
    """
    lines = program.splitlines(keepends=False)

    # Find the method declaration line (contains the method name after lemma/method/function/predicate)
    # Allow leading whitespace for class/module members
    decl_pattern = rf'^\s*(lemma|method|function|predicate)\s+(?:\{{[^}}]*\}}\s*)?{re.escape(method_name)}\b'
    decl_idx = None
    for i, line in enumerate(lines):
        if re.search(decl_pattern, line):
            decl_idx = i
            break

    if decl_idx is None:
        return None

    # Find the opening brace { after the declaration
    brace_idx = None
    for i in range(decl_idx, len(lines)):
        if lines[i].strip() == '{':
            brace_idx = i
            break

    if brace_idx is None:
        return None

    # The declaration line is unique (contains method name), use it as anchor
    # Then use = lines to navigate to the brace
    diff_parts = []
    diff_parts.append(f"@@{lines[decl_idx]}@@")

    # Add = lines for any lines between declaration and brace
    for i in range(decl_idx + 1, brace_idx + 1):
        diff_parts.append(f"= {lines[i]}")

    # Add the body lines
    for body_line in body.splitlines():
        diff_parts.append(f"+ {body_line}")

    return "\n".join(diff_parts)


def get_dafny_errors(program: str, timeout: int = 2) -> str:
    """Run Dafny and capture verification errors."""
    import tempfile
    import os

    # Use temp file since /dev/stdin doesn't work on all systems
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dfy', delete=False) as f:
        f.write(program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["dafny", "verify", f"--verification-time-limit={timeout}", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,  # Subprocess timeout as backup
        )
        # Replace temp path with generic name in output for cleaner errors
        stdout = result.stdout.replace(tmp_path, "program.dfy") if result.stdout else ""
        stderr = result.stderr.replace(tmp_path, "program.dfy") if result.stderr else ""
        return f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    except subprocess.TimeoutExpired:
        return "stdout:\nDafny verification timed out\n\nstderr:\n"
    except Exception as e:
        return f"Error running Dafny: {e}"
    finally:
        os.unlink(tmp_path)


def extract_method_name_from_id(entry_id: str, program: str) -> Optional[str]:
    """
    Extract method/lemma name from VFP entry ID by validating against the program.

    ID format: {file_base}_{method_name}_{type}
    Example: tree_operations_solution_InorderLength_empty -> InorderLength
             tarjan_scc_scc_nonempty_empty -> scc_nonempty
    """
    # Remove type suffix
    for suffix in ["_empty", "_sketch", "_sep"]:
        if entry_id.endswith(suffix):
            entry_id = entry_id[:-len(suffix)]
            break

    # Try different split points and validate against the program
    parts = entry_id.split("_")

    # Try from shortest to longest method name
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[i:])
        # Check if this method exists in the program (allow leading whitespace for class members)
        decl_pattern = rf'^\s*(lemma|method|function|predicate)\s+(?:\{{[^}}]*\}}\s*)?{re.escape(candidate)}\b'
        for line in program.split("\n"):
            if re.search(decl_pattern, line):
                return candidate

    # Fallback: return last part
    return parts[-1] if parts else None


def generate_vfp_examples(
    json_path: Path,
    max_programs: Optional[int] = None,
    skip_dafny: bool = False,
    include_trivial: bool = False,
) -> list[dict]:
    """Generate repair examples from VFP emptied JSON."""
    from patch import apply_text_diff

    if not json_path.exists():
        raise FileNotFoundError(f"VFP JSON file not found: {json_path}")

    with json_path.open() as f:
        entries = json.load(f)

    if max_programs:
        entries = entries[:max_programs]

    examples = []
    skipped_trivial = 0
    skipped_no_method = 0
    skipped_diff_failed = 0
    skipped_bad_diff = 0

    for entry in tqdm(entries, desc="Generating examples"):
        entry_id = entry["id"]
        emptied_code = entry["program"]
        body = entry.get("output", "")

        # Skip trivial entries (empty body) unless requested
        if not body or not body.strip():
            if not include_trivial:
                skipped_trivial += 1
                continue
            # For trivial cases, body is empty - nothing to insert
            body = ""

        # Extract method name from ID (validate against program)
        method_name = extract_method_name_from_id(entry_id, emptied_code)
        if not method_name:
            skipped_no_method += 1
            continue

        # Compute diff directly from the known insertion point
        diff = compute_insertion_diff(emptied_code, method_name, body)
        if diff is None:
            skipped_diff_failed += 1
            continue

        # Verify the diff works by applying it
        try:
            result = apply_text_diff(emptied_code, diff)
            # Check that the body was inserted (result should contain the body)
            if body.strip() and body.strip() not in result:
                skipped_bad_diff += 1
                continue
        except Exception:
            skipped_bad_diff += 1
            continue

        # Get verification errors for emptied version
        if skip_dafny:
            errors = "(Dafny errors would appear here)"
        else:
            tqdm.write(f"  Running Dafny on {entry_id}...")
            errors = get_dafny_errors(emptied_code)

        examples.append({
            "prompt": "repair",
            "arguments": {
                "program": emptied_code,
                "notes": errors,
            },
            "response": diff,
            "outcome": "success",
            "metadata": {
                "source": "vfp",
                "entry_id": entry_id,
                "method_name": method_name,
            }
        })

    if skipped_trivial:
        print(f"Skipped {skipped_trivial} trivial entries (empty body)")
    if skipped_no_method:
        print(f"Skipped {skipped_no_method} entries (couldn't extract method name)")
    if skipped_diff_failed:
        print(f"Skipped {skipped_diff_failed} entries (failed to compute diff)")
    if skipped_bad_diff:
        print(f"Skipped {skipped_bad_diff} entries (bad diff)")

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
    """Generate VFP distillation data."""
    json_path = Path(args.json_path)
    output_path = Path(args.output)

    print(f"Generating examples from {json_path}")

    examples = generate_vfp_examples(
        json_path,
        max_programs=args.max_programs,
        skip_dafny=args.skip_dafny,
        include_trivial=args.include_trivial,
    )

    print(f"Generated {len(examples)} repair examples")

    create_agenda_pickle(examples, output_path)
    print(f"Saved to {output_path}")

    # Print a sample
    if examples:
        print("\n--- Sample example ---")
        sample = examples[0]
        print(f"Entry ID: {sample['metadata']['entry_id']}")
        print(f"Method: {sample['metadata']['method_name']}")
        print(f"Program (first 500 chars):\n{sample['arguments']['program'][:500]}...")
        print(f"\nDiff:\n{sample['response'][:500]}...")


def cmd_verify_diff(args):
    """Verify that generated diffs actually work."""
    from patch import apply_text_diff

    json_path = Path(args.json_path)

    with json_path.open() as f:
        entries = json.load(f)

    entries = entries[:args.max_programs or 10]

    success = 0
    fail = 0
    skip = 0

    for entry in entries:
        entry_id = entry["id"]
        emptied_code = entry["program"]
        body = entry.get("output", "")

        if not body or not body.strip():
            skip += 1
            continue

        method_name = extract_method_name_from_id(entry_id, emptied_code)
        if not method_name:
            skip += 1
            continue

        # Compute diff directly
        diff = compute_insertion_diff(emptied_code, method_name, body)
        if diff is None:
            print(f"SKIP (diff failed): {entry_id}")
            skip += 1
            continue

        try:
            result = apply_text_diff(emptied_code, diff)
            # Check that the body was inserted
            if body.strip() in result:
                success += 1
                print(f"OK: {entry_id}")
            else:
                fail += 1
                print(f"MISMATCH: {entry_id}")
                if args.verbose:
                    print(f"  Body not found in result")
                    print(f"  Diff:\n{diff[:300]}...")
        except Exception as e:
            fail += 1
            print(f"ERROR: {entry_id}: {e}")
            if args.verbose:
                print(f"  Diff:\n{diff[:300]}...")

    print(f"\nResults: {success} OK, {fail} failed, {skip} skipped")


def main():
    parser = argparse.ArgumentParser(description="VFP distillation data generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_p = sub.add_parser("generate", help="Generate VFP distillation data")
    gen_p.add_argument("--json-path", type=str,
                       default="../dafny-vfp-autogen/vfp_autogen_emptied_minimized.json",
                       help="Path to VFP emptied JSON file")
    gen_p.add_argument("--output", "-o", type=str, default="vfp_distill.pkl",
                       help="Output pickle file path")
    gen_p.add_argument("--max-programs", type=int, default=None,
                       help="Maximum number of programs to process")
    gen_p.add_argument("--skip-dafny", action="store_true",
                       help="Skip running Dafny (faster, but no real error messages)")
    gen_p.add_argument("--include-trivial", action="store_true",
                       help="Include trivial entries with empty body")
    gen_p.set_defaults(func=cmd_generate)

    # Verify command
    verify_p = sub.add_parser("verify", help="Verify that generated diffs work")
    verify_p.add_argument("--json-path", type=str,
                          default="../dafny-vfp-autogen/vfp_autogen_emptied_minimized.json",
                          help="Path to VFP emptied JSON file")
    verify_p.add_argument("--max-programs", type=int, default=10,
                          help="Maximum number of programs to verify")
    verify_p.add_argument("--verbose", "-v", action="store_true",
                          help="Show detailed output for failures")
    verify_p.set_defaults(func=cmd_verify_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

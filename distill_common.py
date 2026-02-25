#!/usr/bin/env python3
"""
Shared utilities for distillation and data extraction scripts.

Provides:
- create_agenda_pickle: Write examples to the standard pickle format
- remove_hints: Strip invariants/assertions/decreases from Dafny programs
- compute_text_diff: Compute diffs in the repair-prompt format
- load_verified_programs: Load dafny-program objects filtered by verification status
- get_dafny_errors: Run Dafny and capture verification errors
- DistillExample: TypedDict formalizing the example schema
"""

import difflib
import json
import pickle
import re
import subprocess
import random
from pathlib import Path
from typing import TypedDict

from agenda import Object


def get_content(obj) -> str | None:
    """Extract string content from an object."""
    if obj.content is None:
        return None
    if isinstance(obj.content, bytes):
        return obj.content.decode('utf-8', errors='replace')
    return str(obj.content)


# ---------------------------------------------------------------------------
# Example schema
# ---------------------------------------------------------------------------

class DistillExample(TypedDict, total=False):
    """Schema for a distillation example."""
    prompt: str           # 'implement', 'repair', 'lemma_synth'
    arguments: dict       # task-specific inputs
    response: str         # model output / ground truth
    outcome: str          # 'success', 'goal_unproven', 'fail', 'error'
    metadata: dict        # provenance info


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def create_agenda_pickle(
    examples: list[dict],
    output_path: Path,
    prefix: str = "distil/example",
) -> None:
    """
    Create a pickle file in the format expected by distill.py / eval scripts.

    Args:
        examples: List of example dicts (DistillExample schema)
        output_path: Where to write the pickle
        prefix: Path prefix for objects, e.g. "distil/repair", "distil/implement"
    """
    objects = {}
    for i, ex in enumerate(examples):
        path = f"{prefix}_{i:06d}"
        content = json.dumps(ex, ensure_ascii=False).encode("utf-8")
        obj = Object(path=path, type="distil-example", content=content)
        objects[path] = obj

    pickle_data = {
        "objects": objects,
        "tasks": {},
        "status": {},
        "clock": 0,
    }

    output_path = Path(output_path)
    with output_path.open("wb") as f:
        pickle.dump(pickle_data, f)


# ---------------------------------------------------------------------------
# Pickle loading
# ---------------------------------------------------------------------------

def load_verified_programs(
    pickle_path: Path,
    include_goal_unproven: bool = False,
) -> list[tuple[str, Object]]:
    """
    Load dafny-program objects filtered by verification status.

    Args:
        pickle_path: Path to agenda pickle
        include_goal_unproven: If True, include goal_unproven programs (not just success)

    Returns:
        List of (path, object) tuples for verified programs
    """
    print(f"Loading {pickle_path}...", flush=True)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded pickle ({len(data.get('dataset', {}))} objects)", flush=True)

    valid_statuses = {'success'}
    if include_goal_unproven:
        valid_statuses.add('goal_unproven')

    print(f"Including verification statuses: {valid_statuses}", flush=True)

    verified = []
    for path, obj in data.get('dataset', {}).items():
        if obj.type != 'dafny-program':
            continue
        ver_status = obj.properties.get('verification_status')
        if ver_status in valid_statuses:
            verified.append((path, obj))

    print(f"Found {len(verified)} verified programs", flush=True)
    return verified


# ---------------------------------------------------------------------------
# Hint removal
# ---------------------------------------------------------------------------

def remove_hints(program: str, min_hints: int = 1, lam: float = 2) -> tuple[str, int]:
    """
    Remove hints (invariants, assertions, decreases) from a Dafny program.

    If min_hints is positive, this will remove at least min_hints
    (unless there are fewer than that in the program).

    Otherwise, if min_hints is <= 0, this will always remove all hints.

    Returns:
        (stripped_program, num_hints_removed)
    """
    lines = program.splitlines(keepends=True)
    result_lines = []
    hint_lines = []

    hints_removed = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if (re.match(r'^invariant\b', stripped) or
            re.match(r'^assert[\s(]', stripped) or
            re.match(r'^decreases\b', stripped)):
            hint_lines.append(i)

    if min_hints <= 0:
        min_hints = len(hint_lines)

    lo = min(min_hints, len(hint_lines))

    n_removed = min(len(hint_lines), lo + int(random.expovariate(lam)))
    removed_lines = set(random.sample(hint_lines, k=n_removed))
    result_lines = [l for i, l in enumerate(lines) if i not in removed_lines]

    return ''.join(result_lines), n_removed


# ---------------------------------------------------------------------------
# Text diff
# ---------------------------------------------------------------------------

def compute_text_diff(before: str, after: str) -> str:
    """
    Compute a text diff in the format expected by the repair prompt.

    Format (from patch.py apply_text_diff):
    - @@content@@ anchor (search-forward marker)
    - = line: keep line (find forward, advance cursor)
    - - line: delete line (find forward, delete)
    - + line: add line (insert at cursor)
    """
    before_lines = before.splitlines(keepends=False)
    after_lines = after.splitlines(keepends=False)

    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)

    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        # Add anchor with the line just before the change (if exists)
        if i1 > 0:
            anchor_line = before_lines[i1 - 1]
            diff_parts.append(f"@@{anchor_line}@@")
        else:
            diff_parts.append("@@@@")

        if tag == 'replace':
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


# ---------------------------------------------------------------------------
# Dafny verification
# ---------------------------------------------------------------------------

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

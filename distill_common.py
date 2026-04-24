#!/usr/bin/env python3
"""
Shared utilities for distillation and data extraction scripts.

Provides:
- create_agenda_pickle: Write examples to the standard pickle format
- remove_hints: Strip invariants/assertions/decreases from Dafny programs
- load_verified_programs: Load program objects filtered by verification status
- get_dafny_errors: Run Dafny and capture verification errors
- DistillExample: TypedDict formalizing the example schema
"""

import json
import pickle
import re
import random
from pathlib import Path
from typing import TypedDict

from agenda import Object
from execute import get_dafny_errors


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
    language: str = "dafny",
) -> list[tuple[str, Object]]:
    """
    Load verified program objects filtered by verification status.

    Args:
        pickle_path: Path to agenda pickle
        include_goal_unproven: If True, include goal_unproven programs (not just success)
        language: Formal language (dafny, verus) — used to match object type

    Returns:
        List of (path, object) tuples for verified programs
    """
    program_type = f"{language.lower()}-program"

    print(f"Loading {pickle_path}...", flush=True)
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded pickle ({len(data.get('dataset', {}))} objects)", flush=True)

    valid_statuses = {'success'}
    if include_goal_unproven:
        valid_statuses.add('goal_unproven')

    print(f"Including verification statuses: {valid_statuses}", flush=True)

    verified = []
    for path, obj in data.get('objects', {}).items():
        if not path.startswith('dataset/'):
            continue

        if obj.type != program_type:
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


def remove_hints_verus(program: str, min_hints: int = 1, lam: float = 2) -> tuple[str, int]:
    """
    Remove hints (invariants, assertions, decreases) from a Verus program.

    Verus-specific: handles multi-line `assert(...) by { proof }` and
    `assert forall |...| ... by { proof }` blocks by extending the span to
    cover the matching closing brace.

    Same semantics as remove_hints: removes at least min_hints (or all if
    min_hints <= 0), plus an exponentially-sampled extra.

    Returns:
        (stripped_program, num_hints_removed)
    """
    lines = program.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        is_hint = (
            re.match(r'^invariant\b', stripped)
            or re.match(r'^assert[\s(]', stripped)
            or re.match(r'^decreases\b', stripped)
        )
        if not is_hint:
            i += 1
            continue

        # Extend span across a multi-line `... by { proof }` tail by brace balancing.
        start = i
        end = i
        open_braces = lines[i].count('{') - lines[i].count('}')
        j = i + 1
        while open_braces > 0 and j < len(lines):
            open_braces += lines[j].count('{') - lines[j].count('}')
            end = j
            j += 1
        spans.append((start, end))
        i = end + 1

    if min_hints <= 0:
        min_hints = len(spans)

    lo = min(min_hints, len(spans))
    n_removed = min(len(spans), lo + int(random.expovariate(lam)))

    removed_span_idxs = set(random.sample(range(len(spans)), k=n_removed))
    removed_lines: set[int] = set()
    for idx in removed_span_idxs:
        s, e = spans[idx]
        removed_lines.update(range(s, e + 1))

    result_lines = [l for i, l in enumerate(lines) if i not in removed_lines]
    return ''.join(result_lines), n_removed



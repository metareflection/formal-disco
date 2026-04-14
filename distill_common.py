#!/usr/bin/env python3
"""
Shared utilities for distillation and data extraction scripts.

Provides:
- create_agenda_pickle: Write examples to the standard pickle format
- remove_hints: Strip invariants/assertions/decreases from Dafny programs
- compute_text_diff: Compute diffs in the repair-prompt format
- load_verified_programs: Load program objects filtered by verification status
- get_dafny_errors: Run Dafny and capture verification errors
- DistillExample: TypedDict formalizing the example schema
"""

import difflib
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


# ---------------------------------------------------------------------------
# Text diff
# ---------------------------------------------------------------------------

def compute_text_diff(before: str, after: str, max_context: int = 10) -> str:
    """
    Compute a text diff in the format expected by the repair prompt.

    Format (from patch.py apply_text_diff):
    - @@content@@ anchor (search-forward marker)
    - = line: keep line (find forward, advance cursor)
    - - line: delete line (find forward, delete)
    - + line: add line (insert at cursor)

    Emits up to *max_context* '=' lines before each change block so
    that apply_text_diff can match the anchor + context as a contiguous
    block, disambiguating files with many duplicate lines.
    """
    before_lines = before.splitlines(keepends=False)
    after_lines = after.splitlines(keepends=False)

    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)

    diff_parts = []
    cursor = 0  # tracks how far previous blocks have consumed

    def _pick_context(change_start: int) -> list[str]:
        """Return context lines from before_lines[cursor:change_start].

        Starts with up to ``max_context`` lines immediately before the
        change.  If the resulting block would be ambiguous (i.e. it
        appears more than once between ``cursor`` and the end of the
        file), we extend backwards to include a unique anchor line.
        """
        window = before_lines[cursor:change_start]
        if not window:
            return []

        # Start with the last max_context lines
        ctx = window[-max_context:]

        # Check whether this block appears more than once in the search
        # region (cursor .. end-of-file).  If so, walk backwards to
        # find a unique-enough anchor.
        search_region = before_lines[cursor:]
        n = len(ctx)

        def _count_block(blk: list[str]) -> int:
            """Count how many times blk appears contiguously in search_region."""
            count = 0
            for i in range(len(search_region) - len(blk) + 1):
                if search_region[i:i + len(blk)] == blk:
                    count += 1
            return count

        if _count_block(ctx) > 1:
            # Extend backwards through the equal window looking for a
            # line that makes the block unique.
            for extra in range(1, len(window) - len(ctx) + 1):
                candidate = window[-(n + extra):]
                if _count_block(candidate) == 1:
                    ctx = candidate
                    break
            else:
                # Use the full available window
                ctx = window

        return ctx

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        ctx_lines = _pick_context(i1)

        if ctx_lines:
            # Emit the first non-blank context line as anchor, all
            # others (including blanks) as '=' lines.  A blank first
            # line would produce "@@@@" (empty-anchor syntax) so we
            # skip it for the anchor role and emit it as '='.
            anchor_idx = None
            for ci, cl in enumerate(ctx_lines):
                if cl.strip():
                    anchor_idx = ci
                    break

            if anchor_idx is not None:
                # Emit lines before the anchor as '=' context
                for cl in ctx_lines[:anchor_idx]:
                    diff_parts.append(f"= {cl}")
                diff_parts.append(f"@@{ctx_lines[anchor_idx]}@@")
                for cl in ctx_lines[anchor_idx + 1:]:
                    diff_parts.append(f"= {cl}")
            else:
                # All context lines are blank — emit as @@@@  + '=' lines
                diff_parts.append("@@@@")
                for cl in ctx_lines:
                    diff_parts.append(f"= {cl}")
        elif i1 == 0:
            diff_parts.append("@@@@")
        else:
            diff_parts.append(f"@@{before_lines[i1 - 1]}@@")

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

        cursor = i2

    return "\n".join(diff_parts)



#!/usr/bin/env python3
"""
Filter an agenda pickle by object properties.

Usage:
    # Extract only programs saved by goal_unproven_saver
    python filter_agenda.py agenda-saved.pkl -o saved_only.pkl --prop saved_from_goal_unproven

    # Extract only success programs
    python filter_agenda.py agenda.pkl -o success_only.pkl --prop verification_status=success

    # Keep only programs with non-trivial proofs (decreases clause or longest lemma > 5 lines)
    python filter_agenda.py agenda.pkl -o hard_only.pkl --nontrivial-proof
"""

import argparse
import pickle
import re
from pathlib import Path


def _is_code_line(line: str) -> bool:
    """Return True if the line is not blank and not a pure // comment."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("//")


def _longest_lemma_body(source: str) -> int:
    """Return the length (non-whitespace, non-comment lines) of the longest lemma body."""
    lines = source.split("\n")
    best = 0
    i = 0
    while i < len(lines):
        if re.match(r'\b(lemma|ghost\s+method)\b', lines[i].strip()):
            brace_depth = 0
            found_open = False
            body_lines: list[str] = []
            j = i
            while j < len(lines):
                for ch in lines[j]:
                    if ch == '{':
                        if not found_open:
                            found_open = True
                            brace_depth = 1
                        else:
                            brace_depth += 1
                    elif ch == '}' and found_open:
                        brace_depth -= 1
                        if brace_depth == 0:
                            code_lines = sum(1 for l in body_lines if _is_code_line(l))
                            best = max(best, code_lines)
                            i = j + 1
                            break
                else:
                    if found_open and brace_depth > 0:
                        body_lines.append(lines[j])
                    j += 1
                    continue
                break
            else:
                i = j
                continue
            continue
        i += 1
    return best


def _has_nontrivial_proof(source: str) -> bool:
    """True if source contains a 'decreases' clause or a lemma body longer than 5 code lines."""
    if "decreases" in source:
        return True
    return _longest_lemma_body(source) > 5


def main():
    parser = argparse.ArgumentParser(description="Filter agenda pickle by properties.")
    parser.add_argument('input', type=Path, help="Input pickle file")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Output pickle file")
    parser.add_argument('--prop', action='append', default=[],
                        help="Property filter: 'key' (truthy) or 'key=value'. Can repeat.")
    parser.add_argument('--nontrivial-proof', action='store_true',
                        help="Keep only dafny-program objects with a decreases clause "
                             "or a lemma body longer than 5 non-blank/non-comment lines.")

    args = parser.parse_args()

    if not args.prop and not args.nontrivial_proof:
        parser.error("At least one of --prop or --nontrivial-proof is required.")

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # Parse property filters
    filters = []
    for p in args.prop:
        if '=' in p:
            key, value = p.split('=', 1)
            filters.append((key, value))
        else:
            filters.append((p, None))

    # Filter objects
    filtered = {}
    for path, obj in data['objects'].items():
        # Property filters (AND)
        match = True
        for key, value in filters:
            prop_val = obj.properties.get(key)
            if value is None:
                if not prop_val:
                    match = False
            else:
                if str(prop_val) != value:
                    match = False
        if not match:
            continue

        # Non-trivial proof filter
        if args.nontrivial_proof:
            if getattr(obj, 'type', None) != 'dafny-program':
                continue
            if obj.content is None:
                continue
            source = obj.content.decode('utf-8', errors='replace')
            if not _has_nontrivial_proof(source):
                continue

        filtered[path] = obj

    # Write output with same structure
    out_data = {
        'objects': filtered,
        'tasks': data.get('tasks', {}),
        'status': data.get('status', {}),
        'clock': data.get('clock', 0),
    }

    with open(args.output, 'wb') as f:
        pickle.dump(out_data, f)

    print(f"Filtered {len(filtered)} / {len(data['objects'])} objects → {args.output}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Curate self-contained ACSL-by-Example programs for use as in-context few-shot examples.

For each chosen seed (a `<group>/<name>.c` path under StandardAlgorithms/), this:
  1. Recursively inlines all quoted (non-system) `#include "X"` directives.
     Search paths: <group_dir>, StandardAlgorithms/, StandardAlgorithms/Logic/.
  2. Replaces <limits.h> reference (transitively pulled in by typedefs.h) with
     a tiny C stub providing INT_MIN/INT_MAX/UINT_MAX so the file becomes
     fully self-contained.
  3. Strips header guards from inlined files (so they don't redefine).
  4. Verifies each curated output with `frama-c -wp -wp-prover CVC5,Alt-Ergo`.

Outputs go to language/framac/examples/curated/<name>.c (overwritten).

Usage:
  eval $(opam env)
  python scripts/curate_acsl_examples.py
"""

import re
import subprocess
import sys
from pathlib import Path

SRC = Path("/Users/namin/code/eur/acsl-by-example/StandardAlgorithms")
OUT = Path("/Users/namin/code/eur/formal-disco/language/framac/examples/curated")

# Seeds chosen to span ACSL idioms missing from the model's natural output.
# Each is identified as <group>/<name>; the .c file is the entry point.
SEEDS = [
    "Nonmutating/find",         # multi-behavior + complete/disjoint, terminates, exits, loop variant
    "Mutating/swap",            # \separated, terminates, simple postcondition
    "Mutating/copy",            # \separated, loop variant, ghost-time predicates (Equal{Pre,Here})
    "Numeric/iota",             # loop variant, ranged loop assigns, ensures with quantifier
    "MinMax/clamp",             # multi-behavior, no loops, simple ternary
]

LIMITS_STUB = """
/* Replaces <limits.h> for self-containment. */
#define INT_MAX    2147483647
#define INT_MIN  (-2147483647 - 1)
#define UINT_MAX 4294967295U
""".strip()


def find_include(name: str, group_dir: Path) -> Path | None:
    """Find a quoted include in the corpus's known search paths."""
    for d in (group_dir, SRC, SRC / "Logic"):
        p = d / name
        if p.exists():
            return p
    return None


_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)
_SYS_INCLUDE_RE = re.compile(r'^\s*#include\s+<([^>]+)>\s*$', re.MULTILINE)
_HEADER_GUARD_BEGIN_RE = re.compile(r'^\s*#ifndef\s+\w+_INCLUDED\s*\n\s*#define\s+\w+_INCLUDED\s*\n', re.MULTILINE)
_HEADER_GUARD_END_RE = re.compile(r'\n\s*#endif\s*(/\*[^*]*\*/)?\s*$')


def inline_includes(path: Path, group_dir: Path, seen: set[Path]) -> str:
    """Return the file's content with all quoted includes recursively inlined.
    System includes (<...>) become a comment marker — we'll deal with limits.h separately."""
    if path in seen:
        return f"/* (already inlined: {path.name}) */\n"
    seen.add(path)
    text = path.read_text()

    # Strip header guards (so re-included content doesn't redefine).
    text = _HEADER_GUARD_BEGIN_RE.sub('', text, count=1)
    text = _HEADER_GUARD_END_RE.sub('', text)

    def repl(m: re.Match) -> str:
        name = m.group(1)
        target = find_include(name, group_dir)
        if target is None:
            return f"/* unresolved include: {name} */"
        return f"\n/* === inlined: {name} === */\n" + inline_includes(target, group_dir, seen)

    text = _INCLUDE_RE.sub(repl, text)
    return text


def replace_limits_h(text: str) -> str:
    """Replace <limits.h> with a self-contained stub. Other system includes are left alone but we don't expect any."""
    out_lines = []
    inserted_stub = False
    for line in text.split('\n'):
        m = _SYS_INCLUDE_RE.match(line)
        if m:
            if m.group(1) == 'limits.h':
                if not inserted_stub:
                    out_lines.append(LIMITS_STUB)
                    inserted_stub = True
                # otherwise drop it (already stubbed)
            else:
                out_lines.append(line)  # pass through other system includes
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)


def curate_one(seed: str) -> tuple[Path, str]:
    group, name = seed.split("/")
    c_path = SRC / group / f"{name}.c"
    group_dir = SRC / group
    if not c_path.exists():
        raise FileNotFoundError(c_path)

    text = inline_includes(c_path, group_dir, seen=set())
    text = replace_limits_h(text)

    # Prepend a brief header explaining provenance.
    header = (
        f"/* Curated from acsl-by-example: {seed}.c\n"
        f" * Self-contained: typedefs and helper predicates inlined.\n"
        f" * Verifies with: frama-c -wp -wp-prover CVC5,Alt-Ergo\n"
        f" */\n\n"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{name}.c"
    out_path.write_text(header + text)
    return out_path, text


def verify(path: Path) -> tuple[bool, str]:
    """Run frama-c on the file. Returns (success, last 600 chars of output)."""
    try:
        proc = subprocess.run(
            ["frama-c", "-wp", "-wp-prover", "CVC5,Alt-Ergo,Z3",
             "-wp-timeout", "30", str(path)],
            capture_output=True, text=True, timeout=240,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout after 120s"
    out = proc.stdout + proc.stderr
    proved = re.search(r'Proved goals:\s*(\d+)\s*/\s*(\d+)', out)
    if proc.returncode == 0 and proved:
        p, t = int(proved.group(1)), int(proved.group(2))
        return p == t, f"{p}/{t} proved\n" + out[-400:]
    return False, out[-600:]


def main():
    print(f"Curating {len(SEEDS)} examples to {OUT}")
    for seed in SEEDS:
        out_path, _ = curate_one(seed)
        ok, msg = verify(out_path)
        status = "OK " if ok else "BAD"
        print(f"  [{status}] {seed:30}  -> {out_path.name}")
        if not ok:
            print(f"        {msg.splitlines()[-1] if msg else ''}")
            print(f"        full log saved to {out_path.with_suffix('.log')}")
            out_path.with_suffix('.log').write_text(msg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare Formal Disco vs DafnyBench dataset statistics.

For Formal Disco:
  - Groups programs by parent idea.
  - Keeps only the LARGEST program per idea (by non-empty LOC).
  - Reports K = number of unique ideas, M = remaining "variation" programs.
  - Counts DISTINCT non-empty, non-comment lines across the K kept programs.

For DafnyBench:
  - All 785 .dfy files.
  - Same distinct-line counting.
"""

import json
import re
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def non_empty_lines(content):
    """Return list of stripped non-empty, non-comment lines."""
    out = []
    for line in content.split('\n'):
        s = line.strip()
        if s and not s.startswith('//'):
            out.append(s)
    return out

def analyze(content):
    lines = non_empty_lines(content)
    nel = len(lines)

    # Decls with ensures: scan signature region before first '{'
    def has_ensures(text, pos):
        brace = text.find('{', pos)
        if brace == -1:
            brace = len(text)
        return bool(re.search(r'\bensures\b', text[pos:brace]))

    decl_pat = re.compile(r'\b(method|function|lemma|predicate|function method)\b')
    decls_with_ensures = sum(
        1 for m in decl_pat.finditer(content) if has_ensures(content, m.start())
    )

    n_assert = sum(1 for l in lines if re.match(r'assert\b', l))
    n_inv    = sum(1 for l in lines if re.match(r'invariant\b', l))

    return {
        'nel':                nel,
        'decls_with_ensures': decls_with_ensures,
        'n_assert':           n_assert,
        'n_inv':              n_inv,
        'lines_set':          set(lines),
    }

def sum_stats(stats_list):
    keys = ['nel', 'decls_with_ensures', 'n_assert', 'n_inv']
    total = {k: sum(s[k] for s in stats_list) for k in keys}
    total['distinct_lines'] = len(set().union(*(s['lines_set'] for s in stats_list)))
    return total

# ── Formal Disco ─────────────────────────────────────────────────────────────

print("Loading Formal Disco programs...")
with open('/home/gpoesia/projects/formal-disco/results/agenda-programs.json') as f:
    agenda = json.load(f)

success = [d for d in agenda
           if d.get('properties', {}).get('verification_status') == 'success']

# Group by parent idea, keep largest by non-empty LOC
by_idea: dict[str, dict] = {}
for d in success:
    idea = d['properties'].get('parent_idea', '__no_idea__')
    nel  = len(non_empty_lines(d['content']))
    if idea not in by_idea or nel > len(non_empty_lines(by_idea[idea]['content'])):
        by_idea[idea] = d

K = len(by_idea)                    # unique ideas → K kept programs
M = len(success) - K               # variation programs (same idea, smaller)
print(f"  Total verified:  {len(success):,}")
print(f"  Unique ideas (K): {K:,}")
print(f"  Variations   (M): {M:,}  (same idea, not largest)")

fd_kept = list(by_idea.values())
fd_stats = [analyze(d['content']) for d in fd_kept]
fd = sum_stats(fd_stats)

# ── DafnyBench ───────────────────────────────────────────────────────────────

print("\nLoading DafnyBench programs...")
db_dir = Path('/home/gpoesia/projects/formal-disco/data/DafnyBench/DafnyBench/dataset/ground_truth')
db_files = sorted(db_dir.glob('*.dfy'))
print(f"  {len(db_files)} files")

db_stats = [analyze(p.read_text(errors='replace')) for p in db_files]
db = sum_stats(db_stats)

# ── Print table ───────────────────────────────────────────────────────────────

print()
W = 46
print(f"{'Metric':<{W}} {'Formal Disco (K=' + str(K) + ')':>20} {'DafnyBench':>12} {'Ratio':>7}")
print('-' * (W + 43))

rows = [
    ('Programs counted',             K,              len(db_files)),
    ('Variation programs (excluded)', M,             0),
    ('Non-empty LOC (total)',         fd['nel'],      db['nel']),
    ('Distinct non-empty lines',      fd['distinct_lines'], db['distinct_lines']),
    ('Decls with ensures clause',     fd['decls_with_ensures'], db['decls_with_ensures']),
    ('Assert statements (total)',     fd['n_assert'], db['n_assert']),
    ('Loop invariants (total)',       fd['n_inv'],    db['n_inv']),
]

for label, fd_v, db_v in rows:
    ratio_str = f"{fd_v/db_v:.1f}×" if db_v else "—"
    print(f"{label:<{W}} {fd_v:>20,} {db_v:>12,} {ratio_str:>7}")

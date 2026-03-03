#!/usr/bin/env python3
"""
Diversity metrics for a corpus of Dafny programs.

Usage:
  python diversity.py results/agenda-programs.json --mode longest-per-idea -o report.json
  python diversity.py local-agenda.pkl --mode all -o report.json
  python diversity.py /path/to/folder/ -o report.json
"""

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_DEFAULT = 20

# Dafny keywords kept as-is in templates (not replaced by '*')
DAFNY_KEYWORDS = frozenset({
    # Declarations
    'method', 'function', 'lemma', 'predicate', 'class', 'trait', 'datatype',
    'newtype', 'module', 'import', 'export', 'extends', 'constructor', 'iterator',
    'abstract', 'ghost', 'static', 'protected', 'opaque',
    # Spec clauses
    'requires', 'ensures', 'reads', 'modifies', 'decreases', 'invariant',
    'assert', 'assume', 'reveal',
    # Types
    'int', 'nat', 'real', 'bool', 'char', 'string', 'object',
    'seq', 'set', 'iset', 'multiset', 'map', 'imap', 'array',
    # Expressions / quantifiers
    'forall', 'exists', 'in', 'old', 'fresh', 'allocated', 'unchanged',
    'this', 'null', 'true', 'false',
    # Control flow
    'if', 'then', 'else', 'match', 'case', 'var', 'let', 'calc',
    'while', 'for', 'return', 'returns', 'break', 'continue', 'yield',
    'label', 'new', 'print',
    # Misc
    'by', 'as', 'is', 'witness', 'provides', 'reveals',
})

# Spec-clause line prefixes (used to skip spec lines when finding body braces)
_SPEC_PREFIXES = (
    'requires', 'ensures', 'reads', 'modifies', 'decreases',
    'returns', 'invariant', 'ghost var', 'ghost function',
)

# Common English stopwords filtered out of subject-word counts
# These are often part of Dafny identifier names but carry little semantic info
_SUBJECT_STOPWORDS = frozenset({
    'to', 'is', 'be', 'of', 'in', 'at', 'on', 'by', 'or', 'an',
    'as', 'it', 'do', 'no', 'up', 'if', 'so', 'the', 'and', 'not',
    'for', 'are', 'was', 'has', 'had', 'can', 'may', 'all', 'any',
    'new', 'get', 'set', 'add', 'run', 'use', 'put', 'his', 'her',
    'its', 'our', 'you', 'they', 'but', 'yet', 'nor', 'own', 'out',
    'off', 'via', 'too', 'per', 'due', 'from', 'into', 'with', 'this',
    'that', 'then', 'than', 'when', 'also', 'each', 'both', 'more',
    'such', 'some', 'only', 'same', 'very', 'well', 'just', 'over',
    'after', 'before', 'while', 'where', 'which', 'there', 'their',
    'these', 'those', 'about', 'other', 'first', 'last', 'into',
})


# ---------------------------------------------------------------------------
# Utility: comment removal
# ---------------------------------------------------------------------------

def remove_comments(source: str) -> str:
    """Remove // line comments and /* */ block comments."""
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'//[^\n]*', '', source)
    return source


# ---------------------------------------------------------------------------
# Utility: identifier splitting (camelCase + snake_case → words)
# ---------------------------------------------------------------------------

_CAMEL_RE1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')

def split_identifier(name: str) -> list[str]:
    """
    Split camelCase/PascalCase/snake_case identifier into lowercase words.

    Examples:
      'AddWebPart'  -> ['add', 'web', 'part']
      'find_max'    -> ['find', 'max']
      'URLParser'   -> ['url', 'parser']
    """
    # Strip trailing apostrophes (Dafny primed variables like x')
    name = name.rstrip("'")
    # Insert spaces at camelCase boundaries
    s = _CAMEL_RE1.sub(r'\1 \2', name)
    s = _CAMEL_RE2.sub(r'\1 \2', s)
    # Split on underscores and whitespace
    parts = re.split(r'[_\s]+', s)
    words = []
    for p in parts:
        p = p.lower()
        # Keep only alphabetic words of length >= 2, skip stopwords
        if p.isalpha() and len(p) >= 2 and p not in _SUBJECT_STOPWORDS:
            words.append(p)
    return words


# ---------------------------------------------------------------------------
# Template generation: replace non-keyword identifiers with '*'
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_\']*\b')


def make_template(expr: str) -> str:
    """
    Replace non-keyword identifiers with '*', keeping Dafny keywords as-is.

    Examples:
      'a < b'          -> '* < *'
      'i == 0'         -> '* == 0'
      'forall x :: ...'-> 'forall * :: ...'
    """
    def repl(m: re.Match) -> str:
        w = m.group(0).rstrip("'")
        return w if w in DAFNY_KEYWORDS else '*'
    return _IDENT_RE.sub(repl, expr).strip()


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_DECL_NAME_RE = re.compile(
    r'\b(?:method|function|lemma|predicate|class|trait|datatype|newtype|constructor|iterator)'
    r'\s+([a-zA-Z_][a-zA-Z0-9_\']*)',
    re.MULTILINE,
)

def extract_decl_names(source: str) -> list[str]:
    """Extract names of all declared functions, methods, classes, etc."""
    return [m.group(1) for m in _DECL_NAME_RE.finditer(source)]


_INV_RE = re.compile(r'^\s*invariant\s+(.+)', re.MULTILINE)
_ASSERT_RE = re.compile(r'^\s*assert\s+(.+)', re.MULTILINE)
_ENSURES_RE = re.compile(r'^\s*ensures\s+(.+)', re.MULTILINE)
_REQUIRES_RE = re.compile(r'^\s*requires\s+(.+)', re.MULTILINE)


def _first_line_stripped(s: str) -> str:
    """Return first line of a possibly multi-line string, stripped of trailing ; and whitespace."""
    return s.split('\n')[0].strip().rstrip(';').rstrip('{').strip()


def extract_invariants(clean: str) -> list[str]:
    return [_first_line_stripped(m.group(1)) for m in _INV_RE.finditer(clean)]


def extract_asserts(clean: str) -> list[str]:
    return [_first_line_stripped(m.group(1)) for m in _ASSERT_RE.finditer(clean)]


def extract_ensures(clean: str) -> list[str]:
    return [_first_line_stripped(m.group(1)) for m in _ENSURES_RE.finditer(clean)]


def extract_requires(clean: str) -> list[str]:
    return [_first_line_stripped(m.group(1)) for m in _REQUIRES_RE.finditer(clean)]


# ---------------------------------------------------------------------------
# Brace matching
# ---------------------------------------------------------------------------

def _find_matching_brace(s: str, open_pos: int) -> int:
    """Return index of closing '}' that matches the '{' at s[open_pos]."""
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


# ---------------------------------------------------------------------------
# Loop skeleton
# ---------------------------------------------------------------------------

_LOOP_KW_RE = re.compile(r'\b(while|for)\b')


def _build_skeleton(s: str, start: int, end: int) -> str:
    """Recursively build a loop skeleton string for s[start:end]."""
    result = []
    i = start
    while i < end:
        m = _LOOP_KW_RE.search(s, i, end)
        if m is None:
            break
        keyword = m.group(1)
        brace_start = s.find('{', m.end(), end)
        if brace_start == -1:
            i = m.end()
            continue
        brace_end = _find_matching_brace(s, brace_start)
        brace_end = min(brace_end, end)
        inner = _build_skeleton(s, brace_start + 1, brace_end)
        if inner:
            result.append(f'{keyword} {{ {inner} }}')
        else:
            result.append(f'{keyword} {{ }}')
        i = brace_end + 1
    return ' '.join(result)


def skeleton_stats(skeleton: Optional[str]) -> tuple[int, int]:
    """Return (n_loops, max_nesting_depth) from a loop skeleton string."""
    if not skeleton:
        return 0, 0
    n_loops = len(_LOOP_KW_RE.findall(skeleton))
    depth = max_depth = 0
    for ch in skeleton:
        if ch == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == '}':
            depth -= 1
    return n_loops, max_depth


# Only `method` declarations can have loops in Dafny (not function/lemma/predicate).
_METHOD_LINE_RE = re.compile(
    r'^\s*(?:(?:ghost|static|protected|abstract|opaque)\s+)*method\b',
)


def extract_method_loop_features(source: str) -> list[dict]:
    """
    For each *method* body in source, return a dict with:
      - 'loop_skeleton': skeleton string or None
      - 'n_loops': int
      - 'max_loop_depth': int

    Functions, lemmas, and predicates are skipped because they cannot
    contain loops in Dafny.
    """
    clean = remove_comments(source)
    lines = clean.split('\n')
    results = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if _METHOD_LINE_RE.match(line):
            brace_line: Optional[int] = None
            brace_col: Optional[int] = None

            stripped_decl = line.rstrip()
            if stripped_decl.endswith('{'):
                brace_line = i
                brace_col = line.rindex('{')
            else:
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    stripped = ln.strip()
                    if not stripped:
                        j += 1
                        continue
                    if _is_spec_line(ln):
                        j += 1
                        continue
                    if '{' in ln:
                        brace_col = ln.index('{')
                        brace_line = j
                    break

            if brace_line is None:
                i += 1
                continue

            full = '\n'.join(lines[brace_line:])
            close = _find_matching_brace(full, brace_col)
            body = full[brace_col + 1:close]

            skeleton = _build_skeleton(body, 0, len(body)) or None
            n_loops, max_depth = skeleton_stats(skeleton)
            results.append({
                'loop_skeleton': skeleton,
                'n_loops': n_loops,
                'max_loop_depth': max_depth,
            })

            i = brace_line + body.count('\n') + 1
        else:
            i += 1

    return results


# ---------------------------------------------------------------------------
# Body size extraction
# ---------------------------------------------------------------------------

_DECL_LINE_RE = re.compile(
    r'^\s*(?:(?:ghost|static|protected|abstract|opaque)\s+)*'
    r'(?:method|function|lemma|predicate|constructor)\b',
)


def _is_spec_line(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(kw) for kw in _SPEC_PREFIXES)


def extract_body_sizes(source: str) -> list[int]:
    """
    Return a list of non-blank line counts, one per method/function/lemma body.

    Handles both inline-brace style ('method Foo() {') and
    standalone-brace style ('method Foo()\\n  requires ...\\n{').
    """
    clean = remove_comments(source)
    lines = clean.split('\n')
    sizes = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if _DECL_LINE_RE.match(line):
            brace_line: Optional[int] = None
            brace_col: Optional[int] = None

            # Check if the body { is on the same declaration line
            # (line ends with '{', possibly preceded by ) and whitespace)
            stripped_decl = line.rstrip()
            if stripped_decl.endswith('{'):
                brace_line = i
                brace_col = line.rindex('{')
            else:
                # Scan subsequent lines, skipping spec clauses and blank lines
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    stripped = ln.strip()
                    if not stripped:
                        j += 1
                        continue
                    if _is_spec_line(ln):
                        j += 1
                        continue
                    # First non-spec, non-blank line after declaration
                    if '{' in ln:
                        brace_col = ln.index('{')
                        brace_line = j
                    break

            if brace_line is None:
                i += 1
                continue

            # Build a string starting from the opening brace
            full = '\n'.join(lines[brace_line:])
            close = _find_matching_brace(full, brace_col)
            body = full[brace_col + 1:close]
            count = sum(1 for ln in body.split('\n') if ln.strip())
            sizes.append(count)

            # Advance past this body
            i = brace_line + body.count('\n') + 1
        else:
            i += 1

    return sizes


# ---------------------------------------------------------------------------
# Quantifier signatures
# ---------------------------------------------------------------------------

_QUANTIFIER_RE = re.compile(r'\b(forall|exists)\b')


def quantifier_signature(expr: str) -> Optional[str]:
    """Return space-joined quantifier keywords in order, or None if none found."""
    found = _QUANTIFIER_RE.findall(expr)
    return ' '.join(found) if found else None


def count_quantifiers(expr: str) -> int:
    return len(_QUANTIFIER_RE.findall(expr))


def count_identifiers(expr: str) -> int:
    return len(_IDENT_RE.findall(expr))


# ---------------------------------------------------------------------------
# Per-program analysis
# ---------------------------------------------------------------------------

def analyze_program(source: str) -> dict:
    """Analyze one Dafny program and return raw extracted features."""
    clean = remove_comments(source)

    # Subject words from declaration names
    decl_names = extract_decl_names(source)
    subject_words: list[str] = []
    for name in decl_names:
        subject_words.extend(split_identifier(name))

    # Annotation constructs (extracted from comment-stripped source)
    invariants = extract_invariants(clean)
    asserts = extract_asserts(clean)
    ensures = extract_ensures(clean)
    requires = extract_requires(clean)

    # Templates: replace identifiers with '*'
    inv_templates = [make_template(s) for s in invariants]
    assert_templates = [make_template(s) for s in asserts]
    ensures_templates = [make_template(s) for s in ensures]
    requires_templates = [make_template(s) for s in requires]

    # Per-method loop features (methods only — Dafny functions can't have loops)
    method_loop_features = extract_method_loop_features(source)
    loop_skeletons = [f['loop_skeleton'] for f in method_loop_features if f['loop_skeleton']]
    n_loops_per_method = [f['n_loops'] for f in method_loop_features]
    max_loop_depth_per_method = [f['max_loop_depth'] for f in method_loop_features]

    # Body sizes (all declarations: methods, functions, lemmas, predicates)
    body_sizes = extract_body_sizes(source)

    # Quantifier signatures from requires + ensures
    spec_clauses = ensures + requires
    quantifier_sigs = [q for clause in spec_clauses if (q := quantifier_signature(clause))]

    # Complexity counts per annotation instance
    n_idents_in_asserts = [count_identifiers(a) for a in asserts]
    n_idents_in_invs = [count_identifiers(inv) for inv in invariants]
    n_quantifiers_per_clause = [
        count_quantifiers(c) for c in spec_clauses if count_quantifiers(c) > 0
    ]

    return {
        'subject_words': subject_words,
        'inv_templates': inv_templates,
        'assert_templates': assert_templates,
        'ensures_templates': ensures_templates,
        'requires_templates': requires_templates,
        'loop_skeletons': loop_skeletons,
        'body_sizes': body_sizes,
        'quantifier_sigs': quantifier_sigs,
        'n_loops_per_method': n_loops_per_method,
        'max_loop_depth_per_method': max_loop_depth_per_method,
        'n_idents_in_asserts': n_idents_in_asserts,
        'n_idents_in_invs': n_idents_in_invs,
        'n_quantifiers_per_clause': n_quantifiers_per_clause,
    }


# ---------------------------------------------------------------------------
# Corpus aggregation and report generation
# ---------------------------------------------------------------------------

def entropy(counter: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def make_metric(counter: Counter, top_k: int) -> dict:
    """Build a diversity metric dict from a counter."""
    total = sum(counter.values())
    return {
        'entropy': round(entropy(counter), 4),
        'n_unique': len(counter),
        'n_total': total,
        'top_k': [{'value': str(v), 'count': c} for v, c in counter.most_common(top_k)],
    }


def make_distribution_metric(counter: Counter, top_k: int) -> dict:
    """Diversity metric plus mean/median, for numeric distributions."""
    values: list = []
    for v, c in counter.items():
        values.extend([v] * c)
    if values:
        values_sorted = sorted(values)
        n = len(values_sorted)
        mean = sum(values_sorted) / n
        median = values_sorted[n // 2]
    else:
        mean = median = 0.0

    m = make_metric(counter, top_k)
    m['mean'] = round(float(mean), 3)
    m['median'] = float(median)
    return m


def aggregate(features_list: list[dict], top_k: int = TOP_K_DEFAULT) -> dict:
    """Aggregate per-program features into a corpus-level diversity report."""

    # --- Diversity counters ---
    subject_words: Counter = Counter()
    inv_templates: Counter = Counter()
    assert_templates: Counter = Counter()
    ensures_templates: Counter = Counter()
    requires_templates: Counter = Counter()
    loop_skeletons: Counter = Counter()
    quantifier_sigs: Counter = Counter()

    # --- Complexity distributions ---
    body_sizes: Counter = Counter()
    n_loops_dist: Counter = Counter()
    max_depth_dist: Counter = Counter()
    n_idents_asserts_dist: Counter = Counter()
    n_idents_invs_dist: Counter = Counter()
    n_quants_per_clause_dist: Counter = Counter()

    for f in features_list:
        subject_words.update(f['subject_words'])
        inv_templates.update(t for t in f['inv_templates'] if t)
        assert_templates.update(t for t in f['assert_templates'] if t)
        ensures_templates.update(t for t in f['ensures_templates'] if t)
        requires_templates.update(t for t in f['requires_templates'] if t)

        loop_skeletons.update(f['loop_skeletons'])
        quantifier_sigs.update(f['quantifier_sigs'])

        body_sizes.update(f['body_sizes'])
        n_loops_dist.update(f['n_loops_per_method'])
        max_depth_dist.update(f['max_loop_depth_per_method'])
        n_idents_asserts_dist.update(f['n_idents_in_asserts'])
        n_idents_invs_dist.update(f['n_idents_in_invs'])
        n_quants_per_clause_dist.update(f['n_quantifiers_per_clause'])

    return {
        'diversity': {
            'subject_words': make_metric(subject_words, top_k),
            'invariant_templates': make_metric(inv_templates, top_k),
            'assert_templates': make_metric(assert_templates, top_k),
            'ensures_templates': make_metric(ensures_templates, top_k),
            'requires_templates': make_metric(requires_templates, top_k),
            'loop_skeletons': make_metric(loop_skeletons, top_k),
            'quantifier_signatures': make_metric(quantifier_sigs, top_k),
        },
        'complexity': {
            'body_size': make_distribution_metric(body_sizes, top_k),
            'n_loops': make_distribution_metric(n_loops_dist, top_k),
            'max_loop_depth': make_distribution_metric(max_depth_dist, top_k),
            'n_idents_in_asserts': make_distribution_metric(n_idents_asserts_dist, top_k),
            'n_idents_in_invs': make_distribution_metric(n_idents_invs_dist, top_k),
            'n_quantifiers_per_clause': make_distribution_metric(n_quants_per_clause_dist, top_k),
        },
    }


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _non_empty_lines(content: str) -> int:
    return sum(1 for ln in content.split('\n') if ln.strip() and not ln.strip().startswith('//'))


def load_from_json(path: Path, mode: str) -> list[str]:
    """
    Load programs from agenda-programs.json format.

    mode='longest-per-idea' : one program per unique parent idea (longest by LOC)
    mode='all-success'      : all programs with verification_status == 'success'
    mode='all'              : all programs regardless of status
    """
    with path.open() as f:
        data = json.load(f)

    if mode == 'all':
        entries = data
    else:
        entries = [d for d in data if d.get('properties', {}).get('verification_status') == 'success']

    print(f"  {len(data)} entries total, {len(entries)} after status filter", file=sys.stderr)

    if mode == 'longest-per-idea':
        by_idea: dict[str, dict] = {}
        for d in entries:
            idea = d['properties'].get('parent_idea', '__no_idea__')
            nel = _non_empty_lines(d['content'])
            if idea not in by_idea or nel > _non_empty_lines(by_idea[idea]['content']):
                by_idea[idea] = d
        programs = [d['content'] for d in by_idea.values()]
        print(f"  {len(programs)} unique ideas (longest-per-idea)", file=sys.stderr)
    else:
        programs = [d['content'] for d in entries]
        print(f"  {len(programs)} programs", file=sys.stderr)

    return programs


def load_from_pickle(path: Path, mode: str) -> list[str]:
    """
    Load programs from an agenda checkpoint pickle.

    Looks at objects with path prefix 'programs/' or 'dataset/'.
    """
    with path.open('rb') as f:
        data = pickle.load(f)

    objects = data.get('objects', {})
    prog_objs = {
        k: v for k, v in objects.items()
        if k.startswith('programs/') or k.startswith('dataset/')
    }
    print(f"  {len(prog_objs)} program objects in pickle", file=sys.stderr)

    def get_content(obj) -> Optional[str]:
        c = getattr(obj, 'content', None)
        if c is None:
            return None
        return c.decode('utf-8', errors='replace') if isinstance(c, bytes) else str(c)

    entries = []
    for k, obj in prog_objs.items():
        content = get_content(obj)
        if not content:
            continue
        props = getattr(obj, 'properties', {}) or {}
        outcome = props.get('verification_outcome')
        parents = getattr(obj, 'parents', []) or []
        idea = parents[0] if parents else None
        nel = _non_empty_lines(content)
        entries.append({'key': k, 'idea': idea, 'nel': nel, 'content': content, 'outcome': outcome})

    if mode in ('all-success', 'longest-per-idea'):
        verified_outcomes = {'SUCCESS', 'GOAL_UNPROVEN'}
        entries = [e for e in entries if e['outcome'] in verified_outcomes]
        print(f"  {len(entries)} verified programs", file=sys.stderr)

    if mode == 'longest-per-idea':
        by_idea: dict[str, dict] = {}
        for e in entries:
            ik = e['idea'] or '__no_idea__'
            if ik not in by_idea or e['nel'] > by_idea[ik]['nel']:
                by_idea[ik] = e
        programs = [e['content'] for e in by_idea.values()]
        print(f"  {len(programs)} unique ideas (longest-per-idea)", file=sys.stderr)
    else:
        programs = [e['content'] for e in entries]
        print(f"  {len(programs)} programs", file=sys.stderr)

    return programs


def load_from_folder(path: Path) -> list[str]:
    """Load all .dfy files from a directory."""
    files = sorted(path.glob('*.dfy'))
    print(f"  {len(files)} .dfy files in {path}", file=sys.stderr)
    programs = []
    for f in files:
        try:
            programs.append(f.read_text(errors='replace'))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}", file=sys.stderr)
    return programs


def load_programs(source: str, mode: str) -> tuple[list[str], str]:
    """Load programs from source. Returns (programs, description_string)."""
    path = Path(source)

    if path.is_dir():
        programs = load_from_folder(path)
        desc = f"folder:{source}"
    elif path.suffix == '.pkl':
        programs = load_from_pickle(path, mode)
        desc = f"pickle:{path.name}:{mode}"
    elif path.suffix == '.json':
        programs = load_from_json(path, mode)
        desc = f"json:{path.name}:{mode}"
    else:
        raise ValueError(
            f"Unrecognised source type for '{source}'. "
            "Expected a .pkl file, a .json file, or a directory."
        )

    return programs, desc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Compute diversity metrics for a corpus of Dafny programs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        'source',
        help='Path to a .pkl agenda checkpoint, a .json programs file, or a folder of .dfy files',
    )
    ap.add_argument(
        '--mode',
        choices=['longest-per-idea', 'all-success', 'all'],
        default='longest-per-idea',
        help=(
            'For .pkl/.json sources: which programs to include. '
            '"longest-per-idea" keeps one (longest) program per parent idea from verified programs; '
            '"all-success" includes all verified programs; '
            '"all" includes everything regardless of status. '
            'Ignored for folder sources. (default: longest-per-idea)'
        ),
    )
    ap.add_argument('-o', '--output', default='-', help='Output JSON path (default: stdout)')
    ap.add_argument('--top-k', type=int, default=TOP_K_DEFAULT, help='Top-k entries in report')
    ap.add_argument('--name', default=None, help='Human-readable dataset name for the report')
    args = ap.parse_args()

    print(f"Loading programs from {args.source} ...", file=sys.stderr)
    programs, desc = load_programs(args.source, args.mode)

    if not programs:
        print("No programs found – nothing to analyse.", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {len(programs)} programs ...", file=sys.stderr)
    features_list = []
    for i, prog in enumerate(programs):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(programs)}", file=sys.stderr)
        features_list.append(analyze_program(prog))

    print("Aggregating ...", file=sys.stderr)
    report = aggregate(features_list, top_k=args.top_k)
    report['source'] = desc
    report['name'] = args.name or desc
    report['n_programs'] = len(programs)

    output_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output == '-':
        print(output_json)
    else:
        Path(args.output).write_text(output_json)
        print(f"Report written to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

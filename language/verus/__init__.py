"""Verus language backend implementing verification, complexity metrics, and diversity features.

Verification is performed by invoking `verus` on a temporary file.

Metrics are computed with regex-based analyses adapted for Verus (Rust with
verification annotations), after stripping comments.
"""

import hashlib
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from .. import LanguageBackend, VerificationOutcome, VerificationOutput
from .prompt import VerusPromptBuilder


VERUS_KEYWORDS = frozenset({
    # Declarations
    'fn', 'proof', 'spec', 'struct', 'enum', 'trait', 'impl', 'mod', 'use',
    'pub', 'const', 'static', 'type', 'where', 'crate', 'super', 'self',
    # Spec clauses
    'requires', 'ensures', 'recommends', 'decreases', 'invariant',
    'assert', 'assume', 'reveal', 'reveal_with_fuel',
    # Types
    'int', 'nat', 'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
    'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
    'bool', 'char', 'str', 'String',
    'Vec', 'Seq', 'Set', 'Map', 'Multiset',
    # Expressions / quantifiers
    'forall', 'exists', 'choose', 'trigger',
    'true', 'false',
    # Control flow
    'if', 'else', 'match', 'let', 'mut', 'ref',
    'while', 'for', 'loop', 'return', 'break', 'continue',
    'as', 'in', 'move', 'unsafe',
    # Verus-specific
    'verus', 'open', 'closed', 'Ghost', 'Tracked', 'ghost', 'tracked',
    'by', 'via', 'when',
})

_SPEC_PREFIXES = (
    'requires', 'ensures', 'recommends', 'decreases',
    'invariant', 'opens_invariants',
)

_CAMEL_RE1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_IDENT_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
_DECL_NAME_RE = re.compile(
    r'\b(?:(?:pub\s+)?(?:proof|spec|open\s+spec|closed\s+spec)?\s*fn'
    r'|struct|enum|trait|impl|type)'
    r'\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    re.MULTILINE,
)
_INV_RE = re.compile(r'^\s*invariant\b\s*(.+)', re.MULTILINE)
_ASSERT_RE = re.compile(r'^\s*assert\s*\((.+)', re.MULTILINE)
_ENSURES_RE = re.compile(r'^\s*ensures\b\s*(.+)', re.MULTILINE)
_REQUIRES_RE = re.compile(r'^\s*requires\b\s*(.+)', re.MULTILINE)
_QUANTIFIER_RE = re.compile(r'\b(forall|exists)\b')
_LOOP_KW_RE = re.compile(r'\b(while|for|loop)\b')

# Matches function declarations (regular fn, proof fn, spec fn)
_FN_LINE_RE = re.compile(
    r'^\s*(?:pub\s+)?(?:(?:proof|spec|open\s+spec|closed\s+spec)\s+)?fn\b',
)
# Matches only exec/proof fn (not spec fn) — these can have loops
_EXEC_FN_LINE_RE = re.compile(
    r'^\s*(?:pub\s+)?(?:proof\s+)?fn\b',
)


def _remove_comments(source: str) -> str:
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'//[^\n]*', '', source)
    return source


class _NLTKTools:
    """Lazy-loaded NLTK utilities for subject-word extraction."""

    def __init__(self):
        self._stopwords = None
        self._lemmatizer = None

    def _ensure_loaded(self):
        if self._stopwords is None:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            self._stopwords = set(stopwords.words('english'))
            self._lemmatizer = WordNetLemmatizer()

    def lemmatize_subject_words(self, words: list[str]) -> list[str]:
        """POS-tag words, keep nouns and verbs, lemmatize, and drop stopwords."""
        self._ensure_loaded()
        from nltk import pos_tag
        if not words:
            return []
        tagged = pos_tag(words)
        result = []
        for word, tag in tagged:
            if tag.startswith('NN'):
                result.append(self._lemmatizer.lemmatize(word, 'n'))
            elif tag.startswith('VB'):
                lemma = self._lemmatizer.lemmatize(word, 'v')
                if lemma not in self._stopwords:
                    result.append(lemma)
        return result


_nltk = _NLTKTools()


def _split_identifier(name: str) -> list[str]:
    """Split a camelCase/PascalCase/snake_case identifier into lowercase words."""
    s = _CAMEL_RE1.sub(r'\1 \2', name)
    s = _CAMEL_RE2.sub(r'\1 \2', s)
    parts = re.split(r'[_\s]+', s)
    return [p.lower() for p in parts if p.isalpha() and len(p) >= 2]


def _make_template(expr: str) -> str:
    """Replace non-keyword identifiers with '*', preserving Verus keywords."""
    def repl(m: re.Match) -> str:
        w = m.group(0)
        return w if w in VERUS_KEYWORDS else '*'
    return _IDENT_RE.sub(repl, expr).strip()


def _first_line_stripped(s: str) -> str:
    return s.split('\n')[0].strip().rstrip(';').rstrip(',').rstrip('{').strip()


def _is_spec_line(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(kw) for kw in _SPEC_PREFIXES)


def _is_spec_or_continuation(line: str, fn_indent: int) -> bool:
    """Check if a line is a spec clause or an indented continuation of one.

    In Verus, spec clauses like `requires` often span multiple lines:
        fn foo()
            requires
                x > 0,
                y > 0,
            ensures
                result >= 0,
        {
    Lines indented deeper than the fn declaration that don't contain a `{`
    (or where `{` is inside the continuation) are treated as spec continuations.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if _is_spec_line(line):
        return True
    # A line indented more than the fn and not containing '{' at low indentation
    # is likely a spec continuation
    line_indent = len(line) - len(line.lstrip())
    if line_indent > fn_indent and '{' not in stripped:
        return True
    return False


def _find_matching_brace(s: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(s) - 1


def _build_skeleton(s: str, start: int, end: int) -> str:
    """Recursively build a loop-structure skeleton string for s[start:end]."""
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
        brace_end = min(_find_matching_brace(s, brace_start), end)
        inner = _build_skeleton(s, brace_start + 1, brace_end)
        result.append(f'{keyword} {{ {inner} }}' if inner else f'{keyword} {{ }}')
        i = brace_end + 1
    return ' '.join(result)


def _skeleton_stats(skeleton: Optional[str]) -> tuple[int, int]:
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


def _find_body_brace(lines: list[str], decl_line: int) -> tuple[int | None, int | None]:
    """Find the opening brace of a fn body, skipping multi-line spec clauses.

    Returns (brace_line, brace_col) or (None, None).
    """
    line = lines[decl_line]
    if line.rstrip().endswith('{'):
        return decl_line, line.rindex('{')

    fn_indent = len(line) - len(line.lstrip())
    j = decl_line + 1
    while j < len(lines):
        ln = lines[j]
        if _is_spec_or_continuation(ln, fn_indent):
            j += 1
            continue
        if '{' in ln:
            return j, ln.index('{')
        break
    return None, None


def _extract_fn_loop_features(source: str) -> list[dict]:
    """Return loop feature dicts for each exec/proof fn body in source.

    Spec functions are skipped because they cannot contain loops.
    """
    clean = _remove_comments(source)
    lines = clean.split('\n')
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match exec fn or proof fn, but skip spec fn
        if _EXEC_FN_LINE_RE.match(line) and not re.match(r'^\s*(?:pub\s+)?spec\b', line):
            brace_line, brace_col = _find_body_brace(lines, i)
            if brace_line is None:
                i += 1
                continue
            full = '\n'.join(lines[brace_line:])
            close = _find_matching_brace(full, brace_col)
            body = full[brace_col + 1:close]
            skeleton = _build_skeleton(body, 0, len(body)) or None
            n_loops, max_depth = _skeleton_stats(skeleton)
            results.append({'loop_skeleton': skeleton, 'n_loops': n_loops, 'max_loop_depth': max_depth})
            i = brace_line + body.count('\n') + 1
        else:
            i += 1
    return results


def _extract_body_sizes(source: str) -> list[int]:
    """Return non-blank line counts for each fn body."""
    clean = _remove_comments(source)
    lines = clean.split('\n')
    sizes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FN_LINE_RE.match(line):
            brace_line, brace_col = _find_body_brace(lines, i)
            if brace_line is None:
                i += 1
                continue
            full = '\n'.join(lines[brace_line:])
            close = _find_matching_brace(full, brace_col)
            body = full[brace_col + 1:close]
            sizes.append(sum(1 for ln in body.split('\n') if ln.strip()))
            i = brace_line + body.count('\n') + 1
        else:
            i += 1
    return sizes


class VerusBackend(LanguageBackend):
    """Language backend for Verus (Rust) programs.

    Verification uses the `verus` binary.
    Most metrics are computed with simple regex-based heuristics.
    """

    _COMPLEXITY_METRICS = frozenset({
        'body_sizes', 'n_loops_per_fn', 'n_idents_in_asserts', 'n_idents_in_invs',
    })

    _FEATURE_METRICS = frozenset({
        'subject_words', 'invariant_templates', 'assert_templates',
        'ensures_templates', 'requires_templates', 'loop_skeletons',
    })

    def __init__(self, verus_binary: str | None = None, verus_root: str | None = None) -> None:
        self._verus_binary = verus_binary or os.environ.get("VERUS_BINARY", "verus")
        self._verus_root = verus_root or os.environ.get("VERUS_ROOT") or None

    @property
    def file_extension(self) -> str:
        return 'rs'

    @property
    def declaration_keywords(self) -> list[str]:
        return ['fn', 'proof fn', 'spec fn', 'struct', 'enum',
                'trait', 'invariant', 'assert']

    @property
    def prompt_builder(self) -> VerusPromptBuilder:
        return VerusPromptBuilder()

    @property
    def complexity_metrics(self) -> frozenset[str]:
        return self._COMPLEXITY_METRICS

    @property
    def feature_metrics(self) -> frozenset[str]:
        return self._FEATURE_METRICS

    def strip(self, program: 'Program') -> 'Program':
        clean = _remove_comments(str(program))
        lines = [ln for ln in clean.split('\n') if ln.strip()]
        from language import Program
        return Program('\n'.join(lines), program.language, program.name)

    def verify(self, program: 'Program', timeout: float = 120) -> VerificationOutput:
        home = os.environ.get("HOME", "/tmp")
        tmp_dir = os.path.join(home, "tmp", "formal-disco", "verus")
        key = hashlib.md5(str(program).encode("utf-8")).hexdigest()
        work_dir = os.path.join(tmp_dir, key)
        os.makedirs(work_dir, exist_ok=True)

        tmp_file = os.path.join(work_dir, "ex.rs")

        try:
            source = str(program)
            # Allow legacy benchmarks that lack decreases clauses to verify
            # on post-April-2025 Verus (PR #1545 made them mandatory).
            if "#![verifier::exec_allows_no_decreases_clause]" not in source:
                source = "#![verifier::exec_allows_no_decreases_clause]\n" + source
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(source)

            env = os.environ.copy()
            if self._verus_root:
                env["VERUS_ROOT"] = self._verus_root

            result = subprocess.run(
                [self._verus_binary, "--crate-type", "lib", tmp_file],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            stdout = result.stdout
            stderr = result.stderr
            status = result.returncode
            combined = stdout + stderr

            if status == 0 and "0 errors" in combined:
                outcome = VerificationOutcome.SUCCESS
            elif "postcondition" in combined or "precondition" in combined:
                outcome = VerificationOutcome.GOAL_UNPROVEN
            else:
                outcome = VerificationOutcome.FAIL

            return VerificationOutput(
                outcome=outcome,
                status=status,
                stdout=stdout,
                stderr=stderr,
            )

        except subprocess.TimeoutExpired:
            return VerificationOutput(
                outcome=VerificationOutcome.FAIL,
                status=-1,
                stdout="",
                stderr="timeout",
            )

    def verify_batch(self, programs: list, timeout: float = 120, max_procs: int = 16) -> list[VerificationOutput]:
        results = [None] * len(programs)
        with ThreadPoolExecutor(max_workers=max_procs) as executor:
            future_to_index = {
                executor.submit(self.verify, program, timeout): i
                for i, program in enumerate(programs)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = VerificationOutput(
                        outcome=VerificationOutcome.FAIL, status=-1, stdout="", stderr=str(e)
                    )
        return results

    def complexity(self, program: 'Program') -> dict[str, Any]:
        source = str(program)
        clean = _remove_comments(source)

        asserts = [_first_line_stripped(m.group(1)) for m in _ASSERT_RE.finditer(clean)]
        invariants = [_first_line_stripped(m.group(1)) for m in _INV_RE.finditer(clean)]

        fn_loop_features = _extract_fn_loop_features(source)

        return {
            'body_sizes': _extract_body_sizes(source),
            'n_loops_per_fn': [f['n_loops'] for f in fn_loop_features],
            'n_idents_in_asserts': [len(_IDENT_RE.findall(a)) for a in asserts],
            'n_idents_in_invs': [len(_IDENT_RE.findall(inv)) for inv in invariants],
        }

    def feature_sets(self, program: 'Program') -> dict[str, Counter]:
        source = str(program)
        clean = _remove_comments(source)

        decl_names = [m.group(1) for m in _DECL_NAME_RE.finditer(source)]
        raw_words = [w for name in decl_names for w in _split_identifier(name)]
        subject_words = _nltk.lemmatize_subject_words(raw_words)

        invariants = [_first_line_stripped(m.group(1)) for m in _INV_RE.finditer(clean)]
        asserts = [_first_line_stripped(m.group(1)) for m in _ASSERT_RE.finditer(clean)]
        ensures = [_first_line_stripped(m.group(1)) for m in _ENSURES_RE.finditer(clean)]
        requires = [_first_line_stripped(m.group(1)) for m in _REQUIRES_RE.finditer(clean)]

        fn_loop_features = _extract_fn_loop_features(source)
        loop_skeletons = [f['loop_skeleton'] for f in fn_loop_features if f['loop_skeleton']]

        return {
            'subject_words': Counter(subject_words),
            'invariant_templates': Counter(_make_template(s) for s in invariants),
            'assert_templates': Counter(_make_template(s) for s in asserts),
            'ensures_templates': Counter(_make_template(s) for s in ensures),
            'requires_templates': Counter(_make_template(s) for s in requires),
            'loop_skeletons': Counter(loop_skeletons),
        }

"""Dafny language backend implementing verification, complexity metrics, and diversity features.

Verification is performed by invoking `dafny verify` using the execute module.

Metrics are computed with fairly simple regex-based analyses, and custom bracket parsing when
needed, after stripping comments.

Some of the diversity metrics are based on analyzing word distributions.
We focus on verbs and nouns, and use NLTK to canonicalize and classify words.
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from execute import execute
from .. import LanguageBackend, VerificationOutcome, VerificationOutput
from .prompt import DafnyPromptBuilder


DAFNY_KEYWORDS = frozenset({
    'method', 'function', 'lemma', 'predicate', 'class', 'trait', 'datatype',
    'newtype', 'module', 'import', 'export', 'extends', 'constructor', 'iterator',
    'abstract', 'ghost', 'static', 'protected', 'opaque',
    'requires', 'ensures', 'reads', 'modifies', 'decreases', 'invariant',
    'assert', 'assume', 'reveal',
    'int', 'nat', 'real', 'bool', 'char', 'string', 'object',
    'seq', 'set', 'iset', 'multiset', 'map', 'imap', 'array',
    'forall', 'exists', 'in', 'old', 'fresh', 'allocated', 'unchanged',
    'this', 'null', 'true', 'false',
    'if', 'then', 'else', 'match', 'case', 'var', 'let', 'calc',
    'while', 'for', 'return', 'returns', 'break', 'continue', 'yield',
    'label', 'new', 'print',
    'by', 'as', 'is', 'witness', 'provides', 'reveals',
})

_SPEC_PREFIXES = (
    'requires', 'ensures', 'reads', 'modifies', 'decreases',
    'returns', 'invariant', 'ghost var', 'ghost function',
)

_CAMEL_RE1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_IDENT_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_\']*\b')
_DECL_NAME_RE = re.compile(
    r'\b(?:method|function|lemma|predicate|class|trait|datatype|newtype|constructor|iterator)'
    r'\s+([a-zA-Z_][a-zA-Z0-9_\']*)',
    re.MULTILINE,
)
_INV_RE = re.compile(r'^\s*invariant\s+(.+)', re.MULTILINE)
_ASSERT_RE = re.compile(r'^\s*assert\s+(.+)', re.MULTILINE)
_ENSURES_RE = re.compile(r'^\s*ensures\s+(.+)', re.MULTILINE)
_REQUIRES_RE = re.compile(r'^\s*requires\s+(.+)', re.MULTILINE)
_QUANTIFIER_RE = re.compile(r'\b(forall|exists)\b')
_LOOP_KW_RE = re.compile(r'\b(while|for)\b')
_METHOD_LINE_RE = re.compile(
    r'^\s*(?:(?:ghost|static|protected|abstract|opaque)\s+)*method\b',
)
_DECL_LINE_RE = re.compile(
    r'^\s*(?:(?:ghost|static|protected|abstract|opaque)\s+)*'
    r'(?:method|function|lemma|predicate|constructor)\b',
)


def _remove_comments(source: str) -> str:
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'//[^\n]*', '', source)
    return source


class _NLTKTools:
    """Lazy-loaded NLTK utilities for subject-word extraction.

    This is mainly used in analyzing the words that appear in identifiers in the program,
    which in turn we use to compute diversity metrics.

    Loads stopwords, the POS tagger, and the WordNet lemmatizer on first use
    so that importing this module does not immediately trigger I/O.
    """

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
        """POS-tag words, keep nouns and verbs, lemmatize, and drop stopwords.

        Words that are not recognised English nouns or verbs (e.g. adjectives,
        determiners, prepositions) are discarded, which removes most of the
        noise from identifier names without requiring a hand-crafted stopword list.
        """
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
    name = name.rstrip("'")
    s = _CAMEL_RE1.sub(r'\1 \2', name)
    s = _CAMEL_RE2.sub(r'\1 \2', s)
    parts = re.split(r'[_\s]+', s)
    return [p.lower() for p in parts if p.isalpha() and len(p) >= 2]


def _make_template(expr: str) -> str:
    """Replace non-keyword identifiers with '*', preserving Dafny keywords."""
    def repl(m: re.Match) -> str:
        w = m.group(0).rstrip("'")
        return w if w in DAFNY_KEYWORDS else '*'
    return _IDENT_RE.sub(repl, expr).strip()


def _first_line_stripped(s: str) -> str:
    return s.split('\n')[0].strip().rstrip(';').rstrip('{').strip()


def _is_spec_line(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(kw) for kw in _SPEC_PREFIXES)


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


def _extract_method_loop_features(source: str) -> list[dict]:
    """Return loop feature dicts for each method body in source.

    Functions, lemmas, and predicates are skipped because they cannot
    contain loops in Dafny.
    """
    clean = _remove_comments(source)
    lines = clean.split('\n')
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _METHOD_LINE_RE.match(line):
            brace_line = brace_col = None
            if line.rstrip().endswith('{'):
                brace_line, brace_col = i, line.rindex('{')
            else:
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    if not ln.strip():
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
            n_loops, max_depth = _skeleton_stats(skeleton)
            results.append({'loop_skeleton': skeleton, 'n_loops': n_loops, 'max_loop_depth': max_depth})
            i = brace_line + body.count('\n') + 1
        else:
            i += 1
    return results


def _extract_body_sizes(source: str) -> list[int]:
    """Return non-blank line counts for each method/function/lemma body."""
    clean = _remove_comments(source)
    lines = clean.split('\n')
    sizes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _DECL_LINE_RE.match(line):
            brace_line = brace_col = None
            if line.rstrip().endswith('{'):
                brace_line, brace_col = i, line.rindex('{')
            else:
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    if not ln.strip():
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
            sizes.append(sum(1 for ln in body.split('\n') if ln.strip()))
            i = brace_line + body.count('\n') + 1
        else:
            i += 1
    return sizes


class DafnyBackend(LanguageBackend):
    """Language backend for Dafny programs.

    Verification uses `dafny verify`.
    Most metrics are computed with simple regex-based heuristics.
    """

    _COMPLEXITY_METRICS = frozenset({
        'body_sizes', 'n_loops_per_method', 'n_idents_in_asserts', 'n_idents_in_invs',
    })

    _FEATURE_METRICS = frozenset({
        'subject_words', 'invariant_templates', 'assert_templates',
        'ensures_templates', 'requires_templates', 'loop_skeletons',
    })

    @property
    def file_extension(self) -> str:
        return 'dfy'

    @property
    def declaration_keywords(self) -> list[str]:
        return ['lemma', 'function', 'method', 'datatype', 'class',
                'predicate', 'invariant', 'assert']

    @property
    def prompt_builder(self) -> DafnyPromptBuilder:
        return DafnyPromptBuilder()

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

    def verify(self, program: 'Program', timeout: float = 10) -> VerificationOutput:
        result = execute("dafny verify", "dfy", str(program), timeout=timeout)
        out = result.get('out', '')
        log = result.get('log', '')

        if '0 errors' in out:
            outcome = VerificationOutcome.SUCCESS
        elif 'postcondition' in out:
            outcome = VerificationOutcome.GOAL_UNPROVEN
        else:
            outcome = VerificationOutcome.FAIL

        return VerificationOutput(
            outcome=outcome,
            status=result.get('status', -1),
            stdout=out,
            stderr=log,
        )

    def verify_batch(self, programs: list, timeout: float = 10, max_procs: int = 16) -> list[VerificationOutput]:
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

        method_loop_features = _extract_method_loop_features(source)

        return {
            'body_sizes': _extract_body_sizes(source),
            'n_loops_per_method': [f['n_loops'] for f in method_loop_features],
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

        method_loop_features = _extract_method_loop_features(source)
        loop_skeletons = [f['loop_skeleton'] for f in method_loop_features if f['loop_skeleton']]

        return {
            'subject_words': Counter(subject_words),
            'invariant_templates': Counter(_make_template(s) for s in invariants),
            'assert_templates': Counter(_make_template(s) for s in asserts),
            'ensures_templates': Counter(_make_template(s) for s in ensures),
            'requires_templates': Counter(_make_template(s) for s in requires),
            'loop_skeletons': Counter(loop_skeletons),
        }


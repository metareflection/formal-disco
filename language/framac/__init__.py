"""Frama-C language backend implementing verification, complexity metrics, and diversity features.

Verification is performed by invoking `frama-c -wp` on a temporary file.
Programs are C code annotated with ACSL (ANSI/ISO C Specification Language).

Metrics are computed with regex-based analyses adapted for C with ACSL
annotations, after stripping comments (but preserving ACSL annotations for
analysis before stripping).
"""

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from execute import execute
from .. import LanguageBackend, VerificationOutcome, VerificationOutput
from .prompt import FramaCPromptBuilder

_FEATURES_DIR = Path(__file__).parent / 'features'
_LANGUAGE_FEATURES_JSON = _FEATURES_DIR / 'language-features.json'


FRAMAC_KEYWORDS = frozenset({
    # C keywords
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
    'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void',
    'volatile', 'while',
    # C types
    'size_t', 'bool', 'true', 'false',
    # ACSL contract keywords
    'requires', 'ensures', 'assigns', 'assumes', 'behavior', 'complete',
    'disjoint', 'terminates', 'decreases', 'frees', 'allocates',
    # ACSL loop annotation keywords
    'loop', 'invariant', 'variant',
    # ACSL logic keywords
    'predicate', 'logic', 'lemma', 'axiom', 'axiomatic', 'inductive',
    'assert', 'check', 'admit',
    # ACSL built-ins
    'integer', 'real', 'boolean',
    # ACSL quantifiers / logic
    'forall', 'exists', 'nothing', 'result', 'old', 'at',
    'valid', 'valid_read', 'separated', 'null',
    'true', 'false',
})

_CAMEL_RE1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_IDENT_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
_PREDICATE_NAME_RE = re.compile(
    r'predicate\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    re.MULTILINE,
)
_LOGIC_FN_NAME_RE = re.compile(
    r'logic\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    re.MULTILINE,
)

_ACSL_BLOCK_RE = re.compile(r'/\*@(.*?)\*/', re.DOTALL)
_ACSL_LINE_RE = re.compile(r'//@ *(.*)')

_INV_RE = re.compile(r'loop\s+invariant\s+(.+?)\s*;', re.MULTILINE)
_ASSERT_RE = re.compile(r'(?://@ *assert|/\*@ *assert)\s+(.+?)\s*;', re.MULTILINE)
_ENSURES_RE = re.compile(r'ensures\s+(.+?)\s*;', re.MULTILINE)
_REQUIRES_RE = re.compile(r'requires\s+(.+?)\s*;', re.MULTILINE)
_ASSIGNS_RE = re.compile(r'\bassigns\s+(.+?)\s*;', re.MULTILINE)
_VARIANT_RE = re.compile(r'loop\s+variant\s+(.+?)\s*;', re.MULTILINE)
_LOOP_KW_RE = re.compile(r'\b(while|for|do)\b')

# ACSL "lemma-like" logic declarations, counted analogously to Dafny lemmas.
# Per design: a `predicate`/`logic` function is a definition (like a Dafny
# predicate/function) but in ACSL it lives in annotation space alongside
# `lemma`/`axiomatic`, so we bucket all four as lemma-like here.
_LEMMA_LIKE_RE = re.compile(
    r'\b(lemma|predicate|logic|axiomatic|axiom|inductive)\b', re.MULTILINE)

# A C function *definition*: optional qualifiers, one or more return-type
# tokens (incl. typedefs like `value_type`), the name (group 1), a parameter
# list, then an opening brace. The `[^;{}]` in the params and the required `{`
# distinguish definitions from prototypes (`...);`) and control statements
# (`if (...) {` has no return-type token before the name).
_FN_DEF_RE = re.compile(
    r'(?:^|\n)[ \t]*'
    r'(?:(?:static|inline|extern|const)\s+)*'
    r'(?:[A-Za-z_]\w*\s+|\*+\s*)+'
    r'\*?\s*([A-Za-z_]\w*)\s*'
    r'\([^;{}]*\)\s*\{',
    re.MULTILINE,
)
# Names that would match the pattern but are control flow / operators, not fns.
_NON_FN_NAMES = frozenset({
    'if', 'while', 'for', 'switch', 'return', 'sizeof', 'do', 'else', 'case',
})

_PROVED_RE = re.compile(r'Proved goals:\s*(\d+)\s*/\s*(\d+)')
_UNKNOWN_RE = re.compile(r'\[Unknown\]')
_TIMEOUT_RE = re.compile(r'\[Timeout\]')


def _extract_acsl(source: str) -> str:
    """Extract all ACSL annotation text from the source."""
    parts = []
    for m in _ACSL_BLOCK_RE.finditer(source):
        parts.append(m.group(1))
    for m in _ACSL_LINE_RE.finditer(source):
        parts.append(m.group(1))
    return '\n'.join(parts)


def _remove_comments(source: str) -> str:
    """Remove all comments (both regular C and ACSL) from source."""
    source = re.sub(r'/\*.*?\*/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'//[^\n]*', '', source)
    return source


def _remove_c_comments_only(source: str) -> str:
    """Remove regular C comments but preserve ACSL annotations."""
    source = re.sub(r'/\*(?!@).*?\*/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'//(?!@)[^\n]*', '', source)
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
    s = _CAMEL_RE1.sub(r'\1 \2', name)
    s = _CAMEL_RE2.sub(r'\1 \2', s)
    parts = re.split(r'[_\s]+', s)
    return [p.lower() for p in parts if p.isalpha() and len(p) >= 2]


def _make_template(expr: str) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        return w if w in FRAMAC_KEYWORDS else '*'
    return _IDENT_RE.sub(repl, expr).strip()


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


def _extract_fn_bodies(source: str) -> list[str]:
    """Extract function body strings from C source (comments removed)."""
    clean = _remove_comments(source)
    bodies = []
    for m in _FN_DEF_RE.finditer(clean):
        if m.group(1) in _NON_FN_NAMES:
            continue
        abs_brace = m.end() - 1  # _FN_DEF_RE ends at the opening '{'
        close = _find_matching_brace(clean, abs_brace)
        bodies.append(clean[abs_brace + 1:close])
    return bodies


def _extract_loop_features(source: str) -> list[dict]:
    results = []
    for body in _extract_fn_bodies(source):
        skeleton = _build_skeleton(body, 0, len(body)) or None
        n_loops = len(_LOOP_KW_RE.findall(body)) if body else 0
        results.append({'loop_skeleton': skeleton, 'n_loops': n_loops})
    return results


def _first_line_stripped(s: str) -> str:
    """First non-empty line of a (possibly multi-line) annotation expression."""
    return s.strip().split('\n', 1)[0].strip()


def _preceding_acsl(clean: str, fn_start: int) -> str:
    """ACSL contract text immediately preceding a function declaration.

    `clean` has plain C comments removed but ACSL (/*@ */ and //@) preserved.
    Handles both a single /*@ ... */ block and a run of contiguous //@ lines.
    """
    pre = clean[:fn_start].rstrip()
    if pre.endswith('*/'):
        open_idx = pre.rfind('/*@')
        return pre[open_idx + 3:-2] if open_idx != -1 else ''
    acc = []
    for ln in reversed(pre.split('\n')):
        s = ln.strip()
        if s.startswith('//@'):
            acc.append(s[3:])
        else:
            break
    return '\n'.join(reversed(acc))


def _extract_methods(source: str) -> list[dict]:
    """Per-declaration stats: kind ('function' | 'lemma'), body_size, annotations.

    C function definitions are bucketed as 'function'; ACSL logic declarations
    (lemma/predicate/logic/axiomatic/...) as 'lemma'. Operates on plain-comment-
    stripped source so ACSL annotations are preserved and counted.
    """
    clean = _remove_c_comments_only(source)
    results: list[dict] = []

    for m in _FN_DEF_RE.finditer(clean):
        name = m.group(1)
        if name in _NON_FN_NAMES:
            continue
        abs_brace = m.end() - 1  # _FN_DEF_RE ends at the opening '{'
        close = _find_matching_brace(clean, abs_brace)
        body = clean[abs_brace + 1:close]
        body_size = sum(1 for ln in body.split('\n') if ln.strip())
        contract = _preceding_acsl(clean, m.start())
        annotations = (
            len(_REQUIRES_RE.findall(contract))
            + len(_ENSURES_RE.findall(contract))
            + len(_ASSIGNS_RE.findall(contract))
            + len(_INV_RE.findall(body))
            + len(_VARIANT_RE.findall(body))
            + len(_ASSERT_RE.findall(body))
        )
        results.append({'kind': 'function', 'name': name,
                        'body_size': body_size, 'annotations': annotations})

    acsl = _extract_acsl(clean)
    for m in _LEMMA_LIKE_RE.finditer(acsl):
        if m.group(1) == 'axiomatic':
            brace = acsl.find('{', m.end())
            end = _find_matching_brace(acsl, brace) if brace != -1 else acsl.find(';', m.end())
        else:
            end = acsl.find(';', m.end())
        if end == -1:
            end = len(acsl) - 1
        span = acsl[m.start():end + 1]
        body_size = sum(1 for ln in span.split('\n') if ln.strip())
        results.append({'kind': 'lemma', 'body_size': body_size, 'annotations': 0})

    return results


class FramaCBackend(LanguageBackend):
    """Language backend for Frama-C (C + ACSL) programs.

    Verification uses `frama-c -wp` with CVC5 and Alt-Ergo provers.
    Metrics are computed with regex-based heuristics over ACSL annotations.
    """

    _FEATURE_METRICS = frozenset({
        'subject_word', 'annotation_template', 'loop_skeleton',
        'method_body_size', 'lemma_body_size', 'language_features',
        'annotations_per_method',
    })

    def __init__(self, framac_binary: str | None = None) -> None:
        self._framac_binary = framac_binary or os.environ.get("FRAMAC_BINARY", "frama-c")

    @property
    def file_extension(self) -> str:
        return 'c'

    @property
    def declaration_keywords(self) -> list[str]:
        return ['predicate', 'logic', 'lemma', 'axiom', 'axiomatic',
                'requires', 'ensures', 'assigns',
                'loop invariant', 'loop variant', 'loop assigns',
                'assert']

    @property
    def prompt_builder(self) -> FramaCPromptBuilder:
        return FramaCPromptBuilder()

    @property
    def feature_metrics(self) -> frozenset[str]:
        return self._FEATURE_METRICS

    @property
    def features_dir(self) -> Path:
        return _FEATURES_DIR

    @cached_property
    def _language_feature_regexes(self) -> dict[str, re.Pattern]:
        """Compile the regexes in features/language-features.json once."""
        if not _LANGUAGE_FEATURES_JSON.is_file():
            return {}
        with _LANGUAGE_FEATURES_JSON.open('r', encoding='utf-8') as f:
            entries = json.load(f)
        return {e['id']: re.compile(e['regex']) for e in entries}

    def strip(self, program: 'Program') -> 'Program':
        # Remove only plain C comments; ACSL annotations (/*@ */, //@) are the
        # specification and must be preserved (analogous to Dafny requires/
        # ensures), so all metrics stay invariant to this operation.
        clean = _remove_c_comments_only(str(program))
        lines = [ln for ln in clean.split('\n') if ln.strip()]
        from language import Program
        return Program('\n'.join(lines), program.language, program.name)

    def verify(self, program: 'Program', timeout: float = 60) -> VerificationOutput:
        cmd = f"{self._framac_binary} -wp -wp-prover CVC5,Alt-Ergo"
        try:
            source = str(program)
            result = execute(cmd, "c", source, timeout=timeout)
        except RuntimeError as e:
            return VerificationOutput(
                outcome=VerificationOutcome.FAIL,
                status=-1,
                stdout="",
                stderr=str(e),
            )

        out = result.get('out', '')
        log = result.get('log', '')
        combined = out + log
        status = result.get('status', -1)

        if status != 0:
            return VerificationOutput(
                outcome=VerificationOutcome.FAIL,
                status=status,
                stdout=out,
                stderr=log,
            )

        proved_match = _PROVED_RE.search(combined)
        if proved_match:
            proved = int(proved_match.group(1))
            total = int(proved_match.group(2))
            if proved == total:
                outcome = VerificationOutcome.SUCCESS
            else:
                outcome = VerificationOutcome.GOAL_UNPROVEN
        elif 'No proof obligations' in combined or 'no goal' in combined.lower():
            outcome = VerificationOutcome.SUCCESS
        else:
            outcome = VerificationOutcome.FAIL

        return VerificationOutput(
            outcome=outcome,
            status=status,
            stdout=out,
            stderr=log,
        )

    def verify_batch(self, programs: list, timeout: float = 60, max_procs: int = 16) -> list[VerificationOutput]:
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

    def feature_sets(self, program: 'Program') -> dict[str, Counter]:
        source = str(program)
        # Operate on plain-comment-stripped source (ACSL preserved) so every
        # feature is invariant to strip().
        clean = _remove_c_comments_only(source)
        acsl = _extract_acsl(clean)
        methods = _extract_methods(source)

        # subject_word: nouns/verbs lemmatized from declaration identifiers
        # (C functions + ACSL predicates/logic functions).
        c_names = [d['name'] for d in methods if d['kind'] == 'function']
        pred_names = [m.group(1) for m in _PREDICATE_NAME_RE.finditer(acsl)]
        logic_names = [m.group(1) for m in _LOGIC_FN_NAME_RE.finditer(acsl)]
        raw_words = [w for name in c_names + pred_names + logic_names
                     for w in _split_identifier(name)]
        subject_words = _nltk.lemmatize_subject_words(raw_words)

        # annotation_template: requires/ensures/invariant/assert templates,
        # namespaced with the keyword so kinds remain distinguishable.
        annotation_templates: list[str] = []
        for kind, rx, text in (
            ('invariant', _INV_RE, acsl), ('assert', _ASSERT_RE, clean),
            ('ensures', _ENSURES_RE, acsl), ('requires', _REQUIRES_RE, acsl),
        ):
            annotation_templates.extend(
                f'{kind}: {_make_template(_first_line_stripped(m.group(1)))}'
                for m in rx.finditer(text)
            )

        # loop_skeleton: shape of loops nested inside each function body.
        loop_skeletons = [
            f['loop_skeleton']
            for f in _extract_loop_features(source)
            if f['loop_skeleton']
        ]

        # method_body_size / lemma_body_size / annotations_per_method.
        method_body_sizes: list[int] = []
        lemma_body_sizes: list[int] = []
        annotations_per_method: list[int] = []
        for d in methods:
            if d['kind'] == 'function':
                method_body_sizes.append(d['body_size'])
                annotations_per_method.append(d['annotations'])
            else:
                lemma_body_sizes.append(d['body_size'])

        # language_features: per-feature occurrence counts (ACSL/C constructs).
        language_feature_counts: dict[str, int] = {}
        for fid, rx in self._language_feature_regexes.items():
            n = len(rx.findall(clean))
            if n:
                language_feature_counts[fid] = n

        return {
            'subject_word': Counter(subject_words),
            'annotation_template': Counter(annotation_templates),
            'loop_skeleton': Counter(loop_skeletons),
            'method_body_size': Counter(method_body_sizes),
            'lemma_body_size': Counter(lemma_body_sizes),
            'language_features': Counter(language_feature_counts),
            'annotations_per_method': Counter(annotations_per_method),
        }

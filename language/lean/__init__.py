"""Lean 4 language backend implementing verification, complexity metrics, and diversity features.

Verification is performed by running ``lake env lean`` inside a Lake project directory,
configured via the ``LEAN_PROJECT_DIR`` environment variable.

Metrics are computed with regex-based analyses adapted for Lean 4 syntax.
"""

import hashlib
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .. import LanguageBackend, VerificationOutcome, VerificationOutput
from .prompt import LeanPromptBuilder


LEAN_KEYWORDS = frozenset({
    # Declaration keywords
    'theorem', 'lemma', 'def', 'noncomputable', 'partial',
    'structure', 'class', 'instance', 'inductive', 'abbrev',
    'namespace', 'section', 'end', 'variable', 'open', 'import',
    'where', 'with', 'deriving', 'extends', 'mutual',
    'private', 'protected', 'unsafe', 'opaque',
    # Control flow
    'if', 'then', 'else', 'match', 'let', 'have', 'show',
    'do', 'return', 'for', 'while', 'repeat',
    'fun', 'by', 'calc', 'sorry',
    # Logic & types
    'forall', 'exists', 'Type', 'Prop', 'Sort',
    'True', 'False', 'true', 'false', 'And', 'Or', 'Not', 'Iff',
    # Basic types
    'Nat', 'Int', 'Bool', 'String', 'List', 'Array', 'Option',
    'Unit', 'Fin', 'Float', 'Char', 'IO', 'Set', 'Finset',
    # Tactics
    'simp', 'ring', 'omega', 'norm_num', 'decide', 'exact',
    'apply', 'intro', 'intros', 'rfl', 'rw', 'rewrite',
    'cases', 'induction', 'constructor', 'ext', 'funext',
    'field_simp', 'linarith', 'nlinarith', 'aesop', 'tauto',
    'trivial', 'assumption', 'contradiction', 'exfalso',
    'push_neg', 'congr', 'rcases', 'obtain', 'use',
    'specialize', 'generalize', 'revert', 'clear',
    'unfold', 'dsimp', 'norm_cast', 'positivity',
    'at', 'only', 'using', 'suffices', 'refine',
    # Other
    'in', 'of', 'set_option',
})

_CAMEL_RE1 = re.compile(r'([a-z])([A-Z])')
_CAMEL_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_IDENT_RE = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_\']*\b')

_DECL_NAME_RE = re.compile(
    r'\b(?:theorem|lemma|def|structure|class|instance|inductive|abbrev)'
    r'\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)',
    re.MULTILINE,
)

_TACTIC_RE = re.compile(
    r'\b(simp|ring|omega|norm_num|decide|exact|apply|intro|intros|rfl|'
    r'rw|rewrite|cases|induction|constructor|ext|funext|'
    r'field_simp|linarith|nlinarith|aesop|tauto|trivial|'
    r'assumption|contradiction|exfalso|push_neg|congr|'
    r'rcases|obtain|use|specialize|generalize|revert|'
    r'unfold|dsimp|norm_cast|positivity|calc|have|let|show|'
    r'suffices|refine|sorry)\b'
)

_IMPORT_RE = re.compile(r'^\s*import\s+(\S+)', re.MULTILINE)

_PROOF_STRUCTURE_RE = re.compile(r'\b(by|calc|match|induction|cases|rcases|obtain)\b')

# Hypothesis binders: (x : T), [inst : T], {x : T}
_HYPOTHESIS_RE = re.compile(r'[(\[{][^)\]{}]*:[^)\]{}]+[)\]}]')

# Top-level declarations (at most 2 spaces indent, to avoid matching tactic-level have/let)
_TOPLEVEL_DECL_RE = re.compile(
    r'^\s{0,2}'
    r'(?:(?:noncomputable|partial|private|protected|unsafe)\s+)*'
    r'(?:theorem|lemma|def|instance|abbrev|structure|class|inductive)\b'
)

# Theorem/lemma declarations specifically (for hypothesis extraction)
_THEOREM_RE = re.compile(
    r'^\s{0,2}'
    r'(?:(?:noncomputable|partial|private|protected|unsafe)\s+)*'
    r'(?:theorem|lemma)\b'
)

# Type signature: everything between the declaration name and := or by or where
_TYPE_SIG_RE = re.compile(
    r'(?:theorem|lemma|def|abbrev)\s+\S+\s*(.*?)(?::=|:=|where\b|by\b|\Z)',
    re.DOTALL,
)


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


def _remove_comments(source: str) -> str:
    """Remove Lean 4 block comments (/- ... -/) and line comments (--)."""
    source = re.sub(r'/-.*?-/', ' ', source, flags=re.DOTALL)
    source = re.sub(r'--[^\n]*', '', source)
    return source


def _split_identifier(name: str) -> list[str]:
    """Split a camelCase/PascalCase/snake_case identifier into lowercase words."""
    name = name.split('.')[-1]
    name = name.rstrip("'")
    s = _CAMEL_RE1.sub(r'\1 \2', name)
    s = _CAMEL_RE2.sub(r'\1 \2', s)
    parts = re.split(r'[_\s]+', s)
    return [p.lower() for p in parts if p.isalpha() and len(p) >= 2]


def _make_template(expr: str) -> str:
    """Replace non-keyword identifiers with '*', preserving Lean keywords."""
    def repl(m: re.Match) -> str:
        w = m.group(0).rstrip("'")
        return w if w in LEAN_KEYWORDS else '*'
    return _IDENT_RE.sub(repl, expr).strip()


def _extract_decl_bodies(source: str) -> list[str]:
    """Extract the body text of each top-level declaration.

    Uses indentation heuristics: a body continues until the next
    top-level declaration or end of file.
    """
    clean = _remove_comments(source)
    lines = clean.split('\n')

    decl_starts = []
    for i, line in enumerate(lines):
        if _TOPLEVEL_DECL_RE.match(line):
            decl_starts.append(i)

    bodies = []
    for k, start in enumerate(decl_starts):
        body_start = start + 1
        body_end = decl_starts[k + 1] if k + 1 < len(decl_starts) else len(lines)
        body = '\n'.join(lines[body_start:body_end])
        bodies.append(body)
    return bodies


def _extract_body_sizes(source: str) -> list[int]:
    """Return non-blank line counts for each declaration body."""
    return [sum(1 for ln in body.split('\n') if ln.strip())
            for body in _extract_decl_bodies(source)]


def _extract_tactic_counts(source: str) -> list[int]:
    """Count tactic invocations in each declaration body."""
    return [len(_TACTIC_RE.findall(body))
            for body in _extract_decl_bodies(source)]


def _extract_theorem_signatures(source: str) -> list[str]:
    """Extract the signature lines (up to := or by) for each theorem/lemma."""
    clean = _remove_comments(source)
    lines = clean.split('\n')
    sigs = []
    for i, line in enumerate(lines):
        if _THEOREM_RE.match(line):
            sig_lines = [line]
            j = i + 1
            while j < len(lines):
                ln = lines[j]
                stripped = ln.strip()
                if stripped.startswith(':=') or stripped.startswith('by') or stripped == 'where':
                    break
                if _TOPLEVEL_DECL_RE.match(ln):
                    break
                sig_lines.append(ln)
                if ':=' in ln or ln.rstrip().endswith('by') or ln.rstrip().endswith('where'):
                    break
                j += 1
            sigs.append('\n'.join(sig_lines))
    return sigs


class LeanBackend(LanguageBackend):
    """Language backend for Lean 4 programs.

    Verification uses ``lake env lean`` inside a Lake project directory.
    The project directory is read from the ``LEAN_PROJECT_DIR`` environment variable.
    """

    _COMPLEXITY_METRICS = frozenset({
        'body_sizes', 'tactic_counts', 'n_hypotheses', 'n_idents_in_types',
    })

    _FEATURE_METRICS = frozenset({
        'subject_words', 'tactic_usage', 'import_modules',
        'type_templates', 'proof_structures', 'hypothesis_templates',
    })

    def __init__(self, lake_project_dir: str | None = None) -> None:
        self._lake_project_dir = lake_project_dir or os.environ.get('LEAN_PROJECT_DIR')

    @property
    def file_extension(self) -> str:
        return 'lean'

    @property
    def declaration_keywords(self) -> list[str]:
        return ['theorem', 'lemma', 'def', 'structure', 'class',
                'instance', 'inductive', 'abbrev']

    @property
    def prompt_builder(self) -> LeanPromptBuilder:
        return LeanPromptBuilder()

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

    def verify(self, program: 'Program', timeout: float = 60) -> VerificationOutput:
        project_dir = self._lake_project_dir
        if project_dir is None:
            raise ValueError(
                "LeanBackend requires LEAN_PROJECT_DIR to be set "
                "(path to a Lake project with lakefile.toml)"
            )

        program_text = str(program)
        key = hashlib.md5(program_text.encode("utf-8")).hexdigest()
        tmp_dir = os.path.join(project_dir, ".disco-tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file = os.path.join(tmp_dir, f"{key}.lean")

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(program_text)

            result = subprocess.run(
                ["lake", "env", "lean", tmp_file],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout
            stderr = result.stderr
            status = result.returncode

            source_has_sorry = re.search(r'\bsorry\b', program.program) is not None
            if status == 0 and 'sorry' not in stderr and not source_has_sorry:
                outcome = VerificationOutcome.SUCCESS
            elif status == 0:
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
        finally:
            try:
                os.remove(tmp_file)
            except OSError:
                pass

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

    def complexity(self, program: 'Program') -> dict[str, Any]:
        source = str(program)
        sigs = _extract_theorem_signatures(source)

        return {
            'body_sizes': _extract_body_sizes(source),
            'tactic_counts': _extract_tactic_counts(source),
            'n_hypotheses': [len(_HYPOTHESIS_RE.findall(sig)) for sig in sigs],
            'n_idents_in_types': [len(_IDENT_RE.findall(sig)) for sig in sigs],
        }

    def feature_sets(self, program: 'Program') -> dict[str, Counter]:
        source = str(program)
        clean = _remove_comments(source)

        # Subject words from declaration names
        decl_names = [m.group(1) for m in _DECL_NAME_RE.finditer(source)]
        raw_words = [w for name in decl_names for w in _split_identifier(name)]
        subject_words = _nltk.lemmatize_subject_words(raw_words)

        # Tactic usage
        tactics = _TACTIC_RE.findall(clean)

        # Import modules
        imports = _IMPORT_RE.findall(clean)

        # Type templates from theorem/lemma signatures
        sigs = _extract_theorem_signatures(source)
        type_templates = [_make_template(sig) for sig in sigs]

        # Proof structures
        proof_kws = _PROOF_STRUCTURE_RE.findall(clean)

        # Hypothesis templates
        hyp_templates = []
        for sig in sigs:
            for m in _HYPOTHESIS_RE.finditer(sig):
                hyp_templates.append(_make_template(m.group(0)))

        return {
            'subject_words': Counter(subject_words),
            'tactic_usage': Counter(tactics),
            'import_modules': Counter(imports),
            'type_templates': Counter(type_templates),
            'proof_structures': Counter(proof_kws),
            'hypothesis_templates': Counter(hyp_templates),
        }

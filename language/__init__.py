"""Language-agnostic program representation, verification, and prompting.

Supported backends are implemented as sub-packages (e.g. language/dafny/).
Each backend provides:
  - Verification via the language's external tool.
  - Complexity and diversity metrics for programs.
  - A PromptBuilder that constructs LLM prompts for standard tasks
    (implement, repair, extend, idea generation).
"""

import math
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, TypedDict


class VerificationOutcome(Enum):
    """Result of calling a verifier on a program.

    FAIL: Error, possibly unrelated to verification (e.g. syntax error).
    GOAL_UNPROVEN: Current program is correct but proofs are incomplete.
    SUCCESS: No errors.
    """

    FAIL = 0
    GOAL_UNPROVEN = 1
    SUCCESS = 2


@dataclass
class VerificationOutput:
    outcome: VerificationOutcome
    status: int
    stdout: str
    stderr: str


class ChatMessage(TypedDict):
    role: str
    content: str


class PromptBuilder:
    """Constructs LLM chat prompts for the worker tasks.

    Each method returns a list of ChatMessage dicts (role + content) compatible
    with the standard format (e.g., that langchain uses). The current tasks are:
      - idea: propose a new program idea given a repository README.
      - implement: generate a program from a natural-language idea.
      - repair: fix a failing program given verifier output.
      - extend: expand or improve a working program.
    """

    def implement(self, idea: str) -> list[ChatMessage]:
        raise NotImplementedError

    def repair(
        self,
        program: str,
        notes: str,
        example_before: str,
        example_diff: str,
        example_after: str,
    ) -> list[ChatMessage]:
        raise NotImplementedError

    def extend(
        self,
        program: str,
        example_before: str,
        example_diff: str,
        example_after: str,
    ) -> list[ChatMessage]:
        raise NotImplementedError

    def idea(self, repo: str, readme: str) -> list[ChatMessage]:
        raise NotImplementedError

    def initiate(
        self,
        *,
        repo: str,
        readme: str,
        doc_snippets: Optional[list[tuple[str, str]]] = None,
    ) -> list[ChatMessage]:
        """Prompt the model to come up with an idea and implement it in one shot.

        `doc_snippets` is an optional list of (feature_id, text) pairs sampled
        from the backend's documentation. Backends should surface them in the
        prompt as supplementary inspiration for which language constructs to use.
        """
        raise NotImplementedError

    def generate(self, *, repo: str | None = None, readme: str | None = None) -> list[ChatMessage]:
        """Prompt the model to generate a random program, optionally inspired by a README."""
        raise NotImplementedError

    def repair_full(self, *, program: str, notes: str) -> list[ChatMessage]:
        """Prompt the model to produce a fully repaired program (not a diff)."""
        raise NotImplementedError


class LanguageBackend:
    """Implements language-specific operations over programs.

    A backend supports verification, program feature metrics, and LLM prompt
    construction.  Each backend may also expose additional language-specific
    operations used when deriving training examples.

    A program feature metric is a Counter of occurrences of a given feature
    in the program. Keys must be discrete (str, int, bool)
    so that discrete entropy is a meaningful diversity measure.

    Examples of features:
      - Subject words in identifiers (str)
      - Logical templates of assertions/invariants/pre/post-conditions (str)
      - Loop skeletons (str)
      - Per-method body size (int)
      - Number of loops per method (int)
      - Number of identifiers in each assertion or invariant (int)

    These metrics support two consumers:
      1. Diversity: entropy of the pooled Counter across a corpus measures
         how varied the feature is. For ordered (int) features we may
         additionally compute statistics like median/p90 to track how the
         discovery system pushes those values up over time
         (once we do entropy maximization, and diversity pushes towards higher values).
      2. Per-program uniqueness: how rare are this program's features
         relative to the corpus, used to select in-context examples for
         workers, to prioritize things in the agenda, and to do entropy
         maximization via iterative SFT ranking by uniqueness/surprisal.
    """

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Return the PromptBuilder for this language."""
        raise NotImplementedError

    @property
    def feature_metrics(self) -> set[str]:
        """Return the set of program feature metrics supported by this backend."""
        raise NotImplementedError

    @property
    def surprisal_metrics(self) -> frozenset[str]:
        """Subset of feature_metrics that drives entropy-maximizing data
        selection (distill.py surprisal ranking). The remaining metrics are
        still computed for diversity tracking and in-context selection, but
        do not influence SFT example selection.

        Must be a subset of feature_metrics. Defaults to all of them.
        """
        return frozenset(self.feature_metrics)

    @property
    def file_extension(self) -> str:
        """File extension used for programs in this language (e.g. 'dfy', 'rs')."""
        raise NotImplementedError

    @property
    def declaration_keywords(self) -> list[str]:
        """Top-level declaration keywords for this language (e.g. 'method', 'lemma').

        Used for counting how often each kind of declaration appears in a program.
        """
        raise NotImplementedError

    def strip(self, program: 'Program') -> 'Program':
        """Return a new Program with comments and blank lines removed.

        All metrics (features and complexity) should be invariant to this operation.
        """
        raise NotImplementedError

    def verify(self, program: 'Program', timeout: float = 10) -> VerificationOutput:
        """Call the verifier on the given program and return the outcome."""
        raise NotImplementedError

    def verify_batch(self, programs: list['Program'], timeout: float = 10, max_procs: int = 16) -> list[VerificationOutput]:
        """Call the verifier on a batch of programs in parallel and return the outcomes."""
        raise NotImplementedError

    def feature_sets(self, program: 'Program') -> dict[str, Counter[Any]]:
        """Return a dict of feature Counters for the given program.

        Counter keys must be discrete (str, int, bool) — see class docstring.
        All metric names returned must be present in feature_metrics().
        """
        raise NotImplementedError

    @property
    def features_dir(self) -> Optional[Path]:
        """Directory holding per-feature documentation snippets (*.txt).

        Each .txt filename stem is the feature id; its content is a focused
        prose+example snippet describing one language construct. Backends
        that have no such corpus return None.
        """
        return None

    @cached_property
    def doc_snippets(self) -> dict[str, str]:
        """Map from feature id (filename stem) to snippet text."""
        d = self.features_dir
        if d is None or not d.is_dir():
            return {}
        return {p.stem: p.read_text(encoding='utf-8') for p in sorted(d.glob('*.txt'))}

    def sample_doc_snippets(
        self,
        rng: random.Random,
        n: int,
        weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, str]]:
        """Sample n distinct (feature_id, snippet) pairs from this backend's docs.

        With weights=None, samples uniformly without replacement. Otherwise,
        weights[feature_id] is an unnormalized sampling probability (ids absent
        or with non-positive weight are excluded). This is the hook for later
        entropy-maximizing selection — pass weights inversely proportional to
        a feature's current corpus coverage.
        """
        snippets = self.doc_snippets
        if not snippets or n <= 0:
            return []
        ids = list(snippets.keys())
        n = min(n, len(ids))
        if weights is None:
            chosen = rng.sample(ids, n)
        else:
            remaining_ids = ids[:]
            remaining_w = [max(0.0, weights.get(i, 0.0)) for i in remaining_ids]
            chosen = []
            while len(chosen) < n and any(w > 0 for w in remaining_w):
                idx = rng.choices(range(len(remaining_ids)), weights=remaining_w, k=1)[0]
                chosen.append(remaining_ids[idx])
                remaining_ids.pop(idx)
                remaining_w.pop(idx)
        return [(i, snippets[i]) for i in chosen]

    def surprisal(
        self,
        program: 'Program',
        corpus_stats: dict[str, Counter[Any]],
    ) -> dict[str, float]:
        """Per-metric maximum surprisal (in bits) of a program under a corpus.

        For each feature metric, computes the self-information
        -log2(count(v) / total) for every value v the program exhibits, and
        returns the maximum -- i.e., how surprising the program's rarest value
        of that feature is relative to the pooled corpus distribution.

        corpus_stats[metric] is the pooled Counter across the corpus. The
        program is typically itself in the corpus, so any value it has should
        already be counted; if a value is nevertheless unseen, its surprisal
        is +inf. Metrics for which the program has no values are omitted from
        the result.
        """
        feats = self.feature_sets(program)
        result: dict[str, float] = {}
        for metric, counter in feats.items():
            if not counter:
                continue
            pooled = corpus_stats[metric]
            total = sum(pooled.values())
            max_s = 0.0
            for v in counter:
                c = pooled[v]
                if c == 0:
                    max_s = math.inf
                    break
                s = math.log2(total / c)
                if s > max_s:
                    max_s = s
            result[metric] = max_s
        return result


class Language(Enum):
    """Enumeration of supported formal languages."""

    DAFNY = 0
    VERUS = 1
    LEAN = 2
    FRAMAC = 3

    def get_backend(self) -> LanguageBackend:
        if self == Language.DAFNY:
            from .dafny import DafnyBackend
            return DafnyBackend()
        if self == Language.VERUS:
            from .verus import VerusBackend
            return VerusBackend()
        if self == Language.FRAMAC:
            from .framac import FramaCBackend
            return FramaCBackend()
        if self == Language.LEAN:
            from .lean import LeanBackend
            return LeanBackend()
        raise NotImplementedError(f"Unsupported language: {self}")


ANNOTATION_KEYWORDS = ['assert', 'invariant', 'decreases']


class Program:
    """A program in one of the supported languages."""

    def __init__(self, program: str, language: Language, name: str = None):  # noqa
        self.program = program
        self.language = language
        self.name = name
        self.backend = language.get_backend()

    def to_json_obj(self) -> dict:
        """Return a JSON-serializable representation of this object."""

    @staticmethod
    def from_json_obj(json):
        """Inverse of to_json_obj."""
        return Program(json['program'], json['name'])

    def __str__(self):
        return self.program

    def verify(self) -> VerificationOutput:
        """Call the verifier on this program and return the outcome."""
        return self.backend.verify(self)

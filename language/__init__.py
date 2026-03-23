"""Language-agnostic program representation, verification, and prompting.

Supported backends are implemented as sub-packages (e.g. language/dafny/).
Each backend provides:
  - Verification via the language's external tool.
  - Complexity and diversity metrics for programs.
  - A PromptBuilder that constructs LLM prompts for standard tasks
    (implement, repair, extend, idea generation).
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict


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

    def generate(self, *, repo: str | None = None, readme: str | None = None) -> list[ChatMessage]:
        """Prompt the model to generate a random program, optionally inspired by a README."""
        raise NotImplementedError

    def repair_full(self, *, program: str, notes: str) -> list[ChatMessage]:
        """Prompt the model to produce a fully repaired program (not a diff)."""
        raise NotImplementedError


class LanguageBackend:
    """Implements language-specific operations over programs.

    A backend supports verification, program metrics (complexity and diversity),
    and LLM prompt construction.  Each backend may also expose additional
    language-specific operations used when deriving training examples.

    The metrics work as follows.

    - A program complexity metric is a single number associated with the program
      that measures one dimension of its complexity. These can include:
        - Average loops per method
        - Average function/method body size
        - Average assertion/loop invariant complexity (e.g., number of identifiers)
        - Average loop invariants per method
        - Average lemma body length

      As a rule of thumb, ideally these metrics should not be just related to program size
      (e.g., lines of code). A program can be short and complex, or long and trivial.
      We're still exploring what these metrics /should/ be, but the idea is that the discovery
      system will optimize for finding programs maximizing these metrics, so they should
      reflect properties we want to encourage in our synthetic corpus.

    - A program feature metric counts occurrences of certain features in the program,
      and should be useful for measuring diversity, or semantic program similarity.
      These can include things like:
        - Words appearing in method/lemma/datatype/class names
        - Logical templates of assertions, loop invariants, pre/post-conditions
        - Number of assertions per method
        - Lemma body lengths
        - Loop structures (e.g., "for { for {} }", "while {}")

      These metrics will allow us to do two things:
      1. Compare the feature distributions in our synthetic vs a reference corpus
         (e.g., DafnyBench), allowing us to compare diversity. Entropy is a simple metric for this.
      2. For each program, measure its "uniqueness" with respect to the whole corpus.
         This will allow us to select good in-context examples for workers in the distributed system.
         For instance, a program that uses a very unique loop structure or post-condition might be
         selected over programs using extremely common ones.
    """

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Return the PromptBuilder for this language."""
        raise NotImplementedError

    @property
    def complexity_metrics(self) -> set[str]:
        """Return the set of program complexity metrics supported by this backend."""
        raise NotImplementedError

    @property
    def feature_metrics(self) -> set[str]:
        """Return the set of program feature metrics supported by this backend."""
        raise NotImplementedError

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

    def complexity(self, program: 'Program') -> dict[str, Any]:
        """Return a dict of complexity metrics for the given program.

        All keys must be present in complexity_metrics().
        """
        raise NotImplementedError

    def feature_sets(self, program: 'Program') -> dict[str, Counter[Any]]:
        """Return a dict of feature Counters for the given program.

        Intended for measuring corpus diversity and individual program
        uniqueness.  All keys must be present in feature_metrics().
        """
        raise NotImplementedError


class Language(Enum):
    """Enumeration of supported formal languages."""

    DAFNY = 0
    VERUS = 1
    LEAN = 2

    def get_backend(self) -> LanguageBackend:
        if self == Language.DAFNY:
            from .dafny import DafnyBackend
            return DafnyBackend()
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

"""LLM prompt construction for Frama-C (C + ACSL) programs.

FramaCPromptBuilder implements the PromptBuilder interface for the standard
tasks: implement, repair, extend, and idea generation.

Few-shot mode: if `FRAMAC_FEWSHOT` is set in the environment, the builder
loads .c files from `FRAMAC_FEWSHOT_DIR` (default
`language/framac/examples/curated`) and prepends a small randomly-sampled
subset to the system prompt of `implement`, `initiate`, and `generate`.
The curated examples cover ACSL idioms the model rarely emits on its own
(multi-behavior contracts, \\separated, loop variant, terminates/exits).
"""

import os
import random
from pathlib import Path
from typing import Optional

from .. import ChatMessage, PromptBuilder


_DEFAULT_FEWSHOT_DIR = Path(__file__).parent / "examples" / "curated"


def _format_doc_snippets(snippets: Optional[list[tuple[str, str]]]) -> str:
    """Render sampled doc snippets as a single prompt section, or empty if none."""
    if not snippets:
        return ""
    parts = ["Reference snippets for some ACSL/Frama-C language constructs you may find useful:"]
    for fid, text in snippets:
        parts.append(f"\n--- {fid} ---\n{text.strip()}")
    return "\n".join(parts) + "\n\n"


def _load_fewshot_examples() -> list[str]:
    if not os.environ.get("FRAMAC_FEWSHOT"):
        return []
    d = Path(os.environ.get("FRAMAC_FEWSHOT_DIR", str(_DEFAULT_FEWSHOT_DIR)))
    if not d.is_dir():
        return []
    return sorted(p.read_text() for p in d.glob("*.c"))


class FramaCPromptBuilder(PromptBuilder):
    """Builds chat prompts for Frama-C-specific LLM tasks.

    Each method returns a [system, user] ChatMessage list ready to be
    passed to a chat model.
    """

    def __init__(
        self,
        seed_examples: list[str] | None = None,
        examples_per_call: int = 2,
        rng: random.Random | None = None,
    ) -> None:
        # Explicit seeds win; otherwise auto-load from FRAMAC_FEWSHOT env var.
        self._seeds = seed_examples if seed_examples is not None else _load_fewshot_examples()
        self._k = examples_per_call
        self._rng = rng or random.Random()

    def _format_examples(self) -> str:
        if not self._seeds:
            return ""
        k = min(self._k, len(self._seeds))
        chosen = self._rng.sample(self._seeds, k)
        blocks = "\n\n".join(f"```c\n{s.strip()}\n```" for s in chosen)
        return (
            "\n\nHere are example self-contained C programs with ACSL annotations that"
            " verify with Frama-C/WP. Note the use of multi-behavior contracts (behavior/"
            "complete behaviors/disjoint behaviors), terminates/exits clauses, \\separated"
            " for pointer non-aliasing, loop variant for termination, and named predicates."
            " Use these idioms where appropriate.\n\n"
            f"{blocks}"
        )

    def implement(self, *, idea: str) -> list[ChatMessage]:
        system = (
            "You are an expert C programmer who writes formally verified code using"
            " Frama-C and ACSL (ANSI/ISO C Specification Language). Given a short idea"
            " or specification, output a self-contained C program annotated with ACSL"
            " contracts that can be verified by Frama-C's WP plugin.\n"
            "Your program should NOT try to implement the entire idea, which is likely to be"
            " overly ambitious to write in one go.\n"
            "This is just the beginning: you will later be able to extend and improve the program"
            " incrementally.\n"
            "Start with e.g. a few functions at most, or prove a basic lemma, etc. You can also"
            " add comments on ideas to extend the program later.\n"
            "ACSL annotations go inside /*@ ... */ or //@ comments. Include function contracts"
            " (requires, ensures, assigns) and loop annotations (loop invariant, loop assigns,"
            " loop variant) as appropriate.\n"
            "The output must be valid C code with ACSL annotations and verify with"
            " frama-c -wp when possible. Keep this initial program concise."
            f"{self._format_examples()}"
        )
        user = (
            f"Idea/specification:\n{idea}\n\n"
            "Produce only C source code with ACSL annotations as the response."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def repair(
        self,
        *,
        program: str,
        notes: str,
        example_before: str,
        example_diff: str,
        example_after: str,
    ) -> list[ChatMessage]:
        system = (
            "You are an expert Frama-C developer. You will be given a C program with ACSL"
            " annotations that has errors reported by Frama-C's WP plugin. These errors can be"
            " syntax errors, or verification failures (goals marked [Unknown] or [Timeout]).\n"
            "Your job is to repair these errors by emitting a DIFF in a simple, line-based"
            " format.\n\n"
            "Diff format:\n"
            "- Lines starting with '@@' are anchors (search-forward markers). These don't modify"
            " the program, but just start a new 'block' of changes in your patch.\n"
            "- Lines starting with '=' keep that exact line: find it forward and advance the"
            " cursor. You typically only need a few of these after your @@ line to position the"
            " cursor for the actual changes: you don't need to copy much of the original file.\n"
            "- Lines starting with '-' delete that exact line found forward.\n"
            "- Lines starting with '+' add a new line at the current cursor.\n\n"
            "- All diff lines should start with one of the special characters above and a space"
            " following them. Other lines will be completely ignored\n"
            f"Here is an example of a diff:\n\n"
            f"Text before:\n{example_before}\n\n"
            f"Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
            f"Text after:\n{example_after}"
        )
        user = (
            f"Program:\n{program}\n\n"
            f"Notes (Frama-C WP output):\n{notes}\n\n"
            "Your goal is to fix the errors shown above by Frama-C. Note that fixing these errors"
            " might require various kinds of changes, such as fixing syntax, strengthening or"
            " weakening ACSL contracts, adding loop invariants/variants/assigns clauses, adding"
            " assert annotations as proof hints, introducing ghost code or lemma functions,"
            " adding \\valid or \\separated annotations, or other changes.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def extend(
        self,
        *,
        program: str,
        example_before: str,
        example_diff: str,
        example_after: str,
    ) -> list[ChatMessage]:
        system = (
            "You are an expert Frama-C editor. You will be given a C program with ACSL"
            " annotations.\n"
            "Your job is to EXPAND or improve it by generating a diff in the following simple"
            " format:\n\n"
            "- Lines starting with '@@' are anchors (search-forward markers). These don't modify"
            " the program, but just start a new 'block' of changes in your patch.\n"
            "- Lines starting with '=' keep that exact line: find it forward and advance the"
            " cursor.\n"
            "- Lines starting with '-' delete that exact line found forward.\n"
            "- Lines starting with '+' add a new line at the current cursor.\n\n"
            "- All lines should start with one of the special characters above and a space"
            " following them.\n"
            "Output ONLY the diff. No explanations.\n\n"
            f"Here is an example of a diff:\n\n"
            f"Text before:\n{example_before}\n\n"
            f"Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
            f"Text after:\n{example_after}"
        )
        user = (
            f"Current program:\n{program}\n\n"
            "Goal: Expand or improve the program by proposing a diff.\n"
            "Your diff can add new functions with ACSL contracts, new ACSL predicates or logic"
            " functions, lemma functions with ghost code, new loop annotations (invariant,"
            " variant, assigns), assert annotations as proof hints, or strengthen/refine"
            " existing contracts.\n"
            "Your diff should be focused on one goal, which you are free to decide what to"
            " pursue.\n"
            "The overall goal is to generate interesting formally verified C programs for"
            " training AI assistants for a variety of tasks (synthesis, verification, edit)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, *, repo: str | None = None, readme: str | None = None) -> list[ChatMessage]:
        system = (
            "You are an expert C programmer who writes formally verified code using"
            " Frama-C and ACSL. Your task is to generate an interesting, self-contained"
            " C program with ACSL annotations that verifies with Frama-C's WP plugin.\n"
            "The program should include meaningful specifications: preconditions (requires),"
            " postconditions (ensures), assigns clauses, loop invariants, loop variants,"
            " and logic predicates/functions/lemmas as appropriate.\n"
            "Aim for variety: choose an interesting algorithmic or data-structure topic.\n"
            "Output only valid C source code with ACSL annotations."
            f"{self._format_examples()}"
        )
        if repo and readme:
            user = (
                f"Here is a GitHub repository for inspiration:\n\n"
                f"Repository: {repo}\n\n"
                f"README:\n{readme}\n\n"
                "Generate a self-contained C program with ACSL annotations INSPIRED BY this"
                " repository's theme. The repository is most likely unrelated to formal"
                " verification, so freely adapt or reinterpret its theme.\n"
                "Output only C source code with ACSL annotations."
            )
        else:
            user = (
                "Generate a self-contained C program with ACSL annotations on a topic of"
                " your choice. The goal is to produce an interesting, diverse dataset of"
                " formally verified C programs.\n"
                "Output only C source code with ACSL annotations."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def repair_full(self, *, program: str, notes: str) -> list[ChatMessage]:
        system = (
            "You are an expert Frama-C developer. You will be given a C program with ACSL"
            " annotations that has errors reported by Frama-C's WP plugin. These errors can be"
            " syntax errors, or verification failures (goals marked [Unknown] or [Timeout]).\n"
            "Your job is to produce a COMPLETE, CORRECTED version of the program.\n"
            "Output the full repaired program, not a diff or partial fix.\n"
            "The output must be valid C code with correct ACSL annotations."
        )
        user = (
            f"Program:\n{program}\n\n"
            f"Frama-C WP output:\n{notes}\n\n"
            "Produce the full corrected C program with ACSL annotations. Fix all errors"
            " shown above."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def initiate(
        self,
        *,
        repo: str,
        readme: str,
        doc_snippets: Optional[list[tuple[str, str]]] = None,
    ) -> list[ChatMessage]:
        system = (
            "You are an expert C programmer who writes formally verified code using"
            " Frama-C and ACSL. You will receive a GitHub repository name and its README."
            " Your task is to:\n"
            "1. Come up with a concise idea for a C program with ACSL annotations inspired by"
            " the repository's theme. The repository is most likely unrelated to formal"
            " verification, so freely adapt or reinterpret its theme.\n"
            "2. Immediately implement that idea as a self-contained C program with ACSL"
            " annotations.\n\n"
            "Your program should NOT try to implement the entire idea, which is likely to be"
            " overly ambitious to write in one go.\n"
            "Start with e.g. a few functions at most, or prove a basic lemma, etc. You can also"
            " add comments on ideas to extend the program later.\n"
            "Include function contracts (requires, ensures, assigns) and loop annotations"
            " (loop invariant, loop assigns, loop variant) as appropriate.\n"
            "The output must be valid C code with ACSL annotations and verify with"
            " frama-c -wp when possible. Keep this initial program concise.\n\n"
            "Output ONLY C source code with ACSL annotations. Include a brief comment at"
            " the top describing the idea."
            f"{self._format_examples()}"
        )
        user = (
            f"Repository: {repo}\n\n"
            f"README:\n{readme}\n\n"
            f"{_format_doc_snippets(doc_snippets)}"
            "Come up with an idea for a formally verified C program inspired by this repository"
            " and implement it. Output only C source code with ACSL annotations."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def idea(self, *, repo: str, readme: str) -> list[ChatMessage]:
        system = (
            "You are a helpful assistant that proposes short, precise ideas for C programs"
            " with ACSL annotations that can be formally verified using Frama-C's WP plugin."
            " You will receive a repository name and a README, and you must output exactly one"
            " concise idea and high-level specification for a C program with ACSL contracts"
            " INSPIRED BY the repository. The repository is most likely unrelated to formal"
            " verification, so it is OK to adapt or reinterpret the README and repository names"
            " as long as you attempt to keep its broad theme."
        )
        user = (
            f"Repository: {repo}\n\n"
            f"README:\n{readme}\n\n"
            "Task:\n"
            " - Propose one idea for a C program with ACSL annotations that could be verified"
            " with Frama-C.\n"
            " - Keep it simple and self-contained.\n"
            " - Include a short specification: e.g. preconditions, postconditions, what"
            " properties to verify.\n"
            " - Output only the idea/spec"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

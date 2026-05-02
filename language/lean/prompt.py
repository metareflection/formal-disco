"""LLM prompt construction for Lean 4 programs.

LeanPromptBuilder implements the PromptBuilder interface for the standard
tasks: implement, repair, extend, idea generation, and full repair.
"""

from .. import ChatMessage, PromptBuilder


class LeanPromptBuilder(PromptBuilder):
    """Builds chat prompts for Lean 4-specific LLM tasks.

    Each method returns a [system, user] ChatMessage list ready to be
    passed to a chat model.
    """

    def implement(self, *, idea: str) -> list[ChatMessage]:
        """Prompt the model to write a Lean 4 program from a natural-language idea."""
        system = (
            "You are an expert Lean 4 programmer and mathematician. Given a short idea or"
            " specification, output a self-contained Lean 4 file that implements the idea.\n"
            "Your program should NOT try to implement the entire idea, which is likely to be"
            " overly ambitious to write in one go.\n"
            "This is just the beginning: you will later be able to extend and improve the program"
            " incrementally.\n"
            "Start with e.g. a few definitions and a basic theorem, or a simple structure with"
            " one or two instances. You can add comments on ideas to extend later.\n"
            "Include appropriate imports (e.g. from Mathlib if needed).\n"
            "The output must be valid Lean 4 code that compiles without errors or sorry."
            " Keep this initial program concise."
        )
        user = (
            f"Idea/specification:\n{idea}\n\n"
            "Produce only Lean 4 source code as the response."
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
        """Prompt the model to fix a failing Lean 4 program using a diff."""
        system = (
            "You are an expert Lean 4 developer. You will be given a Lean 4 program that has"
            " errors reported by the Lean elaborator/type checker. These errors can be syntactic,"
            " type errors, or failures to prove goals.\n"
            "Your job is to repair these errors by emitting a DIFF in a simple, line-based"
            " format.\n\n"
            "Diff format:\n"
            "- Lines starting with '@@' are anchors (search-forward markers). These don't modify"
            " the program, but just start a new 'block' of changes in your patch.\n"
            "- Lines starting with '=' keep that exact line: find it forward and advance the"
            " cursor. You typically only need a few of these after your @@ line to position the"
            " cursor for the actual changes.\n"
            "- Lines starting with '-' delete that exact line found forward.\n"
            "- Lines starting with '+' add a new line at the current cursor.\n\n"
            "- All diff lines should start with one of the special characters above and a space"
            " following them. Other lines will be completely ignored.\n"
            f"Here is an example of a diff:\n\n"
            f"Text before:\n{example_before}\n\n"
            f"Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
            f"Text after:\n{example_after}"
        )
        user = (
            f"Program:\n{program}\n\n"
            f"Notes (Lean error output):\n{notes}\n\n"
            "Your goal is to fix the errors shown above. Note that fixing these errors"
            " might require various kinds of changes, such as fixing syntax, correcting types,"
            " adding or modifying proof tactics, adding missing imports, introducing helper"
            " lemmas, or other changes.\n"
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
        """Prompt the model to expand or improve a Lean 4 program using a diff."""
        system = (
            "You are an expert Lean 4 editor. You will be given a Lean 4 program.\n"
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
            "Your diff can add new definitions, theorems, lemmas, instances, structures,"
            " or improve existing proofs. You can strengthen specifications, add new properties,"
            " generalize existing results, or introduce new related concepts.\n"
            "Your diff should be focused on one goal, which you are free to decide what to"
            " pursue.\n"
            "The overall goal is to generate interesting Lean 4 programs for training AI"
            " assistants for a variety of tasks (synthesis, verification, proof)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, *, repo: str | None = None, readme: str | None = None) -> list[ChatMessage]:
        """Prompt the model to generate a random Lean 4 program, optionally inspired by a README."""
        system = (
            "You are an expert Lean 4 programmer and mathematician. Your task is to generate an"
            " interesting, self-contained Lean 4 file that compiles without errors.\n"
            "The program should include meaningful definitions and theorems with complete proofs"
            " (no sorry). Include appropriate imports.\n"
            "Aim for variety: choose an interesting mathematical, algorithmic, or"
            " data-structure topic.\n"
            "Output only valid Lean 4 source code."
        )
        if repo and readme:
            user = (
                f"Here is a GitHub repository for inspiration:\n\n"
                f"Repository: {repo}\n\n"
                f"README:\n{readme}\n\n"
                "Generate a self-contained Lean 4 program INSPIRED BY this repository's theme."
                " The repository is most likely unrelated to theorem proving, so freely"
                " adapt or reinterpret its theme into a mathematical or verified-programming"
                " context.\n"
                "Output only Lean 4 source code."
            )
        else:
            user = (
                "Generate a self-contained Lean 4 program on a topic of your choice."
                " The goal is to produce an interesting, diverse dataset of complete verified"
                " Lean 4 programs with proofs.\n"
                "Output only Lean 4 source code."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def repair_full(self, *, program: str, notes: str) -> list[ChatMessage]:
        """Prompt the model to produce a fully repaired Lean 4 program (not a diff)."""
        system = (
            "You are an expert Lean 4 developer. You will be given a Lean 4 program that has"
            " errors reported by the Lean elaborator/type checker. These errors can be syntactic,"
            " type errors, or failures to prove goals.\n"
            "Your job is to produce a COMPLETE, CORRECTED version of the program.\n"
            "Output the full repaired Lean 4 program, not a diff or partial fix.\n"
            "The output must be valid Lean 4 code that compiles without errors."
        )
        user = (
            f"Program:\n{program}\n\n"
            f"Lean error output:\n{notes}\n\n"
            "Produce the full corrected Lean 4 program. Fix all errors shown above."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def idea(self, *, repo: str, readme: str) -> list[ChatMessage]:
        """Prompt the model to propose a Lean 4 program idea inspired by a GitHub README."""
        system = (
            "You are a helpful assistant that proposes short, precise ideas for Lean 4 programs"
            " that can be formally defined and proved. You will receive a repository name and a"
            " README, and you must output exactly one concise idea and high-level specification"
            " for a Lean 4 program INSPIRED BY the repository. The repository is most likely"
            " unrelated to theorem proving, so it is OK to adapt or reinterpret the README"
            " and repository names as long as you attempt to keep its broad theme."
        )
        user = (
            f"Repository: {repo}\n\n"
            f"README:\n{readme}\n\n"
            "Task:\n"
            " - Propose one idea for a Lean 4 program that could be implemented and proved.\n"
            " - Keep it simple and self-contained.\n"
            " - Include a short specification: e.g. what definitions to make, what properties"
            " to prove.\n"
            " - Output only the idea/spec."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

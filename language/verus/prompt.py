"""LLM prompt construction for Verus (Rust) programs.

VerusPromptBuilder implements the PromptBuilder interface for the four
standard tasks: implement, repair, extend, and idea generation.
"""

from .. import ChatMessage, PromptBuilder


class VerusPromptBuilder(PromptBuilder):
    """Builds chat prompts for Verus-specific LLM tasks.

    Each method returns a [system, user] ChatMessage list ready to be
    passed to a chat model.
    """

    def implement(self, *, idea: str, complex_examples: list[str] = ()) -> list[ChatMessage]:
        """Prompt the model to write a Verus program from a natural-language idea."""
        system = (
            "You are an expert Verus programmer. Given a short idea or specification, output a"
            " self-contained Verus program that implements the idea."
            " Your program should NOT try to implement the entire idea, which is likely to be"
            " overly ambitious to write in one go.\n"
            "This is just the beginning: you will later be able to extend and improve the program"
            " incrementally.\n"
            "Start with e.g. a few functions at most, or prove a basic lemma, etc. You can also"
            " add comments on ideas to extend the program later, too.\n"
            "The output must be valid Verus code (Rust with Verus verification annotations) and"
            " compile/verify when possible. Keep this initial program concise."
            + self._format_complex_examples(complex_examples)
        )
        user = (
            f"Idea/specification:\n{idea}\n\n"
            "Produce only Verus source code as the response."
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
        """Prompt the model to fix a failing Verus program using a diff."""
        system = (
            "You are an expert Verus developer. You will be given a Verus program that has errors"
            " pointed out by the Verus verifier. These errors can be syntactic, or failures to"
            " verify the program (i.e. prove post-conditions or verify current"
            " assertions/invariants).\n"
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
            f"Notes (verification output):\n{notes}\n\n"
            "Your goal is to fix the errors shown above by the Verus verifier. Note that fixing"
            " these errors might require various kinds of changes, such as fixing the syntax,"
            " fixing the implementation of a function, adding new proof annotations (e.g."
            " assert, invariant, decreases clauses, proof blocks, etc), introducing new proof"
            " functions that help prove existing assertions, or other changes.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def extend(
        self,
        *,
        program: str,
        example_before: str,
        example_diff: str,
        example_after: str,
        complex_examples: list[str] = (),
    ) -> list[ChatMessage]:
        """Prompt the model to expand or improve a Verus program using a diff."""
        system = (
            "You are an expert Verus editor. You will be given a Verus program.\n"
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
            + self._format_complex_examples(complex_examples)
        )
        user = (
            f"Current program:\n{program}\n\n"
            "Goal: Expand or improve the program by proposing a diff.\n"
            "Your diff can add new functions, proof functions, structs, enums, traits, new"
            " proof annotations (e.g. requires, ensures, assert, invariant, decreases, etc) to"
            " the current program, improve or edit pre/post-conditions, etc.\n"
            "Your diff should be focused on one goal, which you are free to decide what to"
            " pursue.\n"
            "The overall goal is to generate interesting Verus programs for training AI"
            " assistants for a variety of tasks (synthesis, verification, edit)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, *, repo: str | None = None, readme: str | None = None, complex_examples: list[str] = ()) -> list[ChatMessage]:
        """Prompt the model to generate a random Verus program, optionally inspired by a README."""
        system = (
            "You are an expert Verus programmer. Your task is to generate an interesting,"
            " self-contained Verus program that compiles and verifies.\n"
            "The program should include meaningful specifications: preconditions, postconditions,"
            " loop invariants, assertions, or proof functions as appropriate.\n"
            "Aim for variety: choose an interesting algorithmic or data-structure topic.\n"
            "Output only valid Verus source code (Rust with Verus verification annotations)."
            + self._format_complex_examples(complex_examples)
        )
        if repo and readme:
            user = (
                f"Here is a GitHub repository for inspiration:\n\n"
                f"Repository: {repo}\n\n"
                f"README:\n{readme}\n\n"
                "Generate a self-contained Verus program INSPIRED BY this repository's theme."
                " The repository is most likely unrelated to verified programming, so freely"
                " adapt or reinterpret its theme.\n"
                "Output only Verus source code."
            )
        else:
            user = (
                "Generate a self-contained Verus program on a topic of your choice."
                " The goal is to produce an interesting, diverse dataset of complete verified"
                " Verus programs.\n"
                "Output only Verus source code."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def repair_full(self, *, program: str, notes: str) -> list[ChatMessage]:
        """Prompt the model to produce a fully repaired Verus program (not a diff)."""
        system = (
            "You are an expert Verus developer. You will be given a Verus program that has errors"
            " reported by the Verus verifier. These errors can be syntactic, or failures to"
            " verify the program (i.e. prove post-conditions or verify assertions/invariants).\n"
            "Your job is to produce a COMPLETE, CORRECTED version of the program.\n"
            "Output the full repaired Verus program, not a diff or partial fix.\n"
            "The output must be valid Verus code."
        )
        user = (
            f"Program:\n{program}\n\n"
            f"Verifier output:\n{notes}\n\n"
            "Produce the full corrected Verus program. Fix all errors shown above."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def initiate(self, *, repo: str, readme: str, complex_examples: list[str] = ()) -> list[ChatMessage]:
        """Prompt the model to propose an idea and implement it as Verus code in one shot."""
        system = (
            "You are an expert Verus programmer. You will receive a GitHub repository name and"
            " its README. Your task is to:\n"
            "1. Come up with a concise idea for a Verus program inspired by the repository's theme."
            " The repository is most likely unrelated to verified programming, so freely adapt or"
            " reinterpret its theme.\n"
            "2. Immediately implement that idea as a self-contained Verus program.\n\n"
            "Your program should NOT try to implement the entire idea, which is likely to be"
            " overly ambitious to write in one go.\n"
            "Start with e.g. a few functions at most, or prove a basic lemma, etc. You can also"
            " add comments on ideas to extend the program later.\n"
            "The output must be valid Verus code (Rust with Verus verification annotations) and"
            " compile/verify when possible. Keep this initial program concise.\n\n"
            "Output ONLY Verus source code. Include a brief comment at the top of the program"
            " describing the idea."
            + self._format_complex_examples(complex_examples)
        )
        user = (
            f"Repository: {repo}\n\n"
            f"README:\n{readme}\n\n"
            "Come up with an idea for a Verus program inspired by this repository and implement it."
            " Output only Verus source code."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def idea(self, *, repo: str, readme: str) -> list[ChatMessage]:
        """Prompt the model to propose a Verus program idea inspired by a GitHub README."""
        system = (
            "You are a helpful assistant that proposes short, precise ideas for Verus programs"
            " that can be fully specified and verified. You will receive a repository name and a"
            " README, and you must output exactly one concise idea and high-level specification"
            " for a Verus program INSPIRED BY the repository. The repository is most likely"
            " unrelated to verified programming, so it is OK to adapt or reinterpret the README"
            " and repository names as long as you attempt to keep its broad theme."
        )
        user = (
            f"Repository: {repo}\n\n"
            f"README:\n{readme}\n\n"
            "Task:\n"
            " - Propose one idea for a Verus program that could be implemented and verified.\n"
            " - Keep it simple and self-contained.\n"
            " - Include a short specification: e.g. preconditions, postconditions, broadly what"
            " to verify.\n"
            " - Output only the idea/spec"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

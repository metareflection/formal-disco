#!/usr/bin/env python3

"""Prompt formatting and reconstruction.

This module centralizes *all* prompt strings used by workers and by distillation.
"""

from typing import Any, Mapping, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


def system_implement() -> str:
    return (
        "You are an expert Dafny programmer. Given a short idea or specification, output a"
        " self-contained Dafny program that implements the idea."
        "Your program should NOT try to implement the entire idea, which is likely to be overly ambitious to write in one go.\n"
        "This is just the beginning: you will later be able to extend and improve the program incrementally.\n"
        "Start with e.g., a few functions at most, or a very basic class with only a couple of core methods, or prove a basic lemma, etc. You can also add comments on ideas to extend the program later, too.\n"
        "The output must be valid Dafny code and"
        " compile/verify when possible. Keep this initial program concise."
    )


def system_repair(*, example_before: str, example_diff: str, example_after: str) -> str:
    return (
        "You are an expert Dafny developer. You will be given a Dafny program that has errors "
        "pointed out by Dafny. These errors can be syntactic, or failures to verify the program (i.e., prove post-conditions or verify current assertions/invariants).\n"
        "Your job is to repair these errors by emitting a DIFF in a simple, line-based format.\n\n"
        "Diff format:\n"
        "- Lines starting with '@@' are anchors (search-forward markers). These don't modify the program, but just start a new 'block' of changes in your patch.\n"
        "- Lines starting with '=' keep that exact line: find it forward and advance the cursor. You typically only need a few of these after your @@ line to position the cursor for the actual changes: you don't need to copy much of the original file.\n"
        "- Lines starting with '-' delete that exact line found forward.\n"
        "- Lines starting with '+' add a new line at the current cursor.\n\n"
        "- All diff lines should start with one of the special characters above and a space following them. Other lines will be completely ignored\n"
        "Here is an example of a diff:\n\n"
        f"Text before:\n{example_before}\n\n"
        f"Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
        f"Text after:\n{example_after}"
    )


def system_extend(*, example_before: str, example_diff: str, example_after: str) -> str:
    return (
        "You are an expert Dafny editor. You will be given a Dafny program.\n"
        "Your job is to EXPAND or improve it by generating a diff in the following simple format:\n\n"
        "- Lines starting with '@@' are anchors (search-forward markers). These don't modify the program, but just start a new 'block' of changes in your patch.\n"
        "- Lines starting with '=' keep that exact line: find it forward and advance the cursor.\n"
        "- Lines starting with '-' delete that exact line found forward.\n"
        "- Lines starting with '+' add a new line at the current cursor.\n\n"
        "- All lines should start with one of the special characters above and a space following them.\n"
        "Output ONLY the diff. No explanations.\n\nHere is an example of a diff:\n\n"
        f"Text before:\n{example_before}\n\n"
        f"Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
        f"Text after:\n{example_after}"
    )


def system_idea() -> str:
    return (
        "You are a helpful assistant that proposes short, precise ideas for Dafny programs "
        "that can be fully specified and verified. You will receive a repository name and a README, "
        "and you must output exactly one concise idea and high-level specification for a Dafny program "
        "INSPIRED BY the repository. The repository is most likely unrelated to verified programming, so "
        "it is OK to adapt or reinterpret the README and repository names as long as you attempt to "
        "keep its broad theme."
    )


def format_implement_user(*, idea: str) -> str:
    return (
        "Idea/specification:\n{idea}\n\n"
        "Produce only Dafny source code as the response."
    ).format(idea=idea)


def format_repair_user(*, program: str, notes: str) -> str:
    return (
        "Program:\n{program}\n\n"
        "Notes (verification output):\n{notes}\n\n"
        "Your goal is to fix the errors shown above by Dafny. Note that fixing these errors might require various kinds of changes, such as fixing the syntax, fixing the implementation of a method or function, adding new logical annotations (e.g., assertions, invariants, decreases/increases clauses, etc), introducing new lemmas that help prove existing assertions, or other changes.\n"
    ).format(program=program, notes=notes)


def format_extend_user(*, program: str) -> str:
    return (
        "Current program:\n{program}\n\n"
        "Goal: Expand or improve the program by proposing a diff.\n"
        "Your diff can add a new method or classes, new lemmas, tests, logical annotations (e.g., assertions, invariants, decreases, etc) to the current program, improve or edit pre/post-conditions, etc.\n"
        "Your diff should be focused on one goal, which you are free to decide what to pursue.\n"
        "The overall goal is to generate interesting Dafny programs for training AI assistants for a variety of tasks (synthesis, verification, edit)."
    ).format(program=program)


def format_idea_user(*, repo: str, readme: str) -> str:
    return (
        "Repository: {repo}\n\n"
        "README:\n"
        "{readme}\n\n"
        "Task:\n"
        " - Propose one idea for a Dafny program that could be implemented and verified.\n"
        " - Keep it simple and self-contained.\n"
        " - Include a short specification: e.g., preconditions, postconditions, broadly what to verify.\n"
        " - Output only the idea/spec"
    ).format(repo=repo, readme=readme)


def reconstruct_user_prompt(kind: str, arguments: Mapping[str, Any]) -> str:
    """Reconstruct the user message content from a distillation example."""
    kind = str(kind)

    if kind == "implement":
        return format_implement_user(idea=str(arguments.get("idea", "")))

    if kind == "repair":
        return format_repair_user(
            program=str(arguments.get("program", "")),
            notes=str(arguments.get("notes", "")),
        )

    if kind == "extend":
        return format_extend_user(program=str(arguments.get("program", "")))

    if kind == "idea":
        return format_idea_user(
            repo=str(arguments.get("repo", "")),
            readme=str(arguments.get("readme", "")),
        )

    raise ValueError(f"Unknown distill prompt kind: {kind!r}")


def build_chat_messages(
    kind: str,
    arguments: Mapping[str, Any],
    *,
    example_before: str | None = None,
    example_diff: str | None = None,
    example_after: str | None = None,
) -> list[ChatMessage]:
    """Return a full chat prompt (system+user) for a given task kind."""
    kind = str(kind)

    if kind == "implement":
        return [
            {"role": "system", "content": system_implement()},
            {"role": "user", "content": reconstruct_user_prompt(kind, arguments)},
        ]

    if kind == "repair":
        if example_before is None or example_diff is None or example_after is None:
            raise ValueError("repair chat requires example_before/example_diff/example_after")
        return [
            {
                "role": "system",
                "content": system_repair(
                    example_before=example_before,
                    example_diff=example_diff,
                    example_after=example_after,
                ),
            },
            {"role": "user", "content": reconstruct_user_prompt(kind, arguments)},
        ]

    if kind == "extend":
        if example_before is None or example_diff is None or example_after is None:
            raise ValueError("extend chat requires example_before/example_diff/example_after")
        return [
            {
                "role": "system",
                "content": system_extend(
                    example_before=example_before,
                    example_diff=example_diff,
                    example_after=example_after,
                ),
            },
            {"role": "user", "content": reconstruct_user_prompt(kind, arguments)},
        ]

    if kind == "idea":
        return [
            {"role": "system", "content": system_idea()},
            {"role": "user", "content": reconstruct_user_prompt(kind, arguments)},
        ]

    raise ValueError(f"Unknown distill prompt kind: {kind!r}")


def reconstruct_chat_messages(
    kind: str,
    arguments: Mapping[str, Any],
    example_before: str | None = None,
    example_diff: str | None = None,
    example_after: str | None = None,
) -> list[ChatMessage]:
    """build_chat_messages for distillation."""
    return build_chat_messages(
        kind,
        arguments,
        example_before=example_before,
        example_diff=example_diff,
        example_after=example_after,
    )

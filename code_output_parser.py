#!/usr/bin/env python3

import re
from typing import Any

from typing_extensions import override

from langchain_core.output_parsers import BaseOutputParser


class CodeOutputParser(BaseOutputParser[str]):
    """
    Extracts code content from LLM outputs that may be wrapped in fenced code blocks.

    Behavior:
    - If any line matches a language fence like "```<lang>", then find the LAST such fence,
      and return the text between that line and the next closing triple backtick line.
    - If no language fence is present, return the entire output as-is.

    This helps when models wrap outputs in markdown fences like ```dafny ... ```.
    """

    # Match e.g., ```python, ```dafny, ```js, ```diff, etc.
    _lang_fence_re = re.compile(r"^```[A-Za-z0-9_.+-]+\s*$")
    _fence_re = re.compile(r"^```\s*$")

    @override
    def parse(self, text: str, *, partial: bool = False) -> str:  # type: ignore[override]
        lines = text.splitlines(keepends=True)
        has_lang_fence = any(self._lang_fence_re.match(line.strip()) for line in lines)

        if not has_lang_fence:
            return text

        # Find last language fence (opening) and next fence (closing)
        open_idx: int | None = None
        close_idx: int | None = None

        for i, line in enumerate(lines):
            if self._lang_fence_re.match(line.strip()):
                open_idx = i

        if open_idx is None:
            return text

        for j in range(open_idx + 1, len(lines)):
            if lines[j].strip().startswith("```"):
                close_idx = j
                break

        # If no closing fence, return everything after the opening fence
        if close_idx is None:
            return "".join(lines[open_idx + 1 :])

        # Return content between fences (excluding the fences themselves)
        return "".join(lines[open_idx + 1 : close_idx])

    @property
    def _type(self) -> str:
        return "code_output_parser"


def test_passthrough_when_no_language_fence():
    text = "Hello world\nThis is plain output without fences.\n"
    parser = CodeOutputParser()
    assert parser.parse(text) == text


def test_extracts_between_language_fence():
    text = (
        "Some preface text that should be ignored by the parser only when a language fence exists.\n"
        "```dafny\n"
        "method Foo() {}\n"
        "lemma Bar() ensures true {}\n"
        "```\n"
        "Some epilogue that should be ignored.\n"
    )
    parser = CodeOutputParser()
    out = parser.parse(text)
    assert out == "method Foo() {}\nlemma Bar() ensures true {}\n"


def test_extracts_after_open_when_no_closing_fence():
    essential_text = (
        "print('hi')\n"
        "no close fence follows\n"
    )
    text = (
        "header\n"
        "```python\n"
        f"{essential_text}"
    )
    parser = CodeOutputParser()
    out = parser.parse(text)
    assert out == essential_text


def test_prefers_last_block_when_lang_fence_present():
    text = (
        "intro\n"
        "```dafny\n"
        "Dafny content A\n"
        "```\n"
        "```python\n"
        "Python content B\n"
        "```\n"
    )
    parser = CodeOutputParser()
    out = parser.parse(text)
    assert out == "Python content B\n"

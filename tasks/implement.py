#!/usr/bin/env python3
"""
Implement task: generate Dafny code from an idea/specification.

Single-shot evaluation: idea -> LLM generates code -> verify.

Migrated from: eval_implement.py, implementer_distill.py
"""

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser
from dafny import DafnyProgram, VerificationOutcome
from prompt import system_implement, format_implement_user, reconstruct_chat_messages
from tasks import EvaluationTask, load_pickle_source, load_dfy_source

logger = logging.getLogger(__name__)


class ImplementTask(EvaluationTask):
    """Implement Dafny programs from ideas/specifications."""

    name = "implement"
    prompt_type = "implement"

    def __init__(
        self,
        sources: list[dict] | None = None,
        verbose: bool = False,
    ):
        self.sources = sources or []
        self.verbose = verbose
        self._chain = None

    def _ensure_chain(self, llm: Any) -> None:
        if self._chain is not None:
            return
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_implement()),
            ("user", "{user_message}"),
        ])
        self._chain = prompt | llm | CodeOutputParser()

    # ------------------------------------------------------------------
    # (a) Data extraction
    # ------------------------------------------------------------------

    def extract_examples(self, sources: list[dict] | None = None) -> list[dict]:
        """Extract implement examples.

        Pickle sources: load distill examples with prompt_type='implement'.
        Dfy sources: extract idea comments from .dfy files.
        """
        sources = sources or self.sources
        examples = []

        for source in sources:
            src_type = source.get("type", "pickle")

            if src_type == "pickle":
                raw = load_pickle_source({
                    **source,
                    "prompt_types": source.get("prompt_types", ["implement"]),
                })
                for ex in raw:
                    idea = ex.get('arguments', {}).get('idea', '')
                    if not idea:
                        continue
                    examples.append(ex)

            elif src_type == "dfy":
                programs = load_dfy_source(source)
                for name, text in programs:
                    idea = self._extract_idea_prompt(text)
                    if not idea:
                        continue
                    cleaned = self._remove_idea_comment(text)
                    examples.append({
                        "prompt": "implement",
                        "arguments": {"idea": idea},
                        "response": cleaned,
                        "outcome": "success",
                        "metadata": {"source": "dfy", "file_name": name},
                    })

        return examples

    @staticmethod
    def _extract_idea_prompt(program: str) -> str | None:
        """Extract '// Idea prompt: ...' comment from a program."""
        for line in program.splitlines():
            stripped = line.strip()
            if stripped.startswith("// Idea prompt:"):
                return stripped[len("// Idea prompt:"):].strip()
        return None

    @staticmethod
    def _remove_idea_comment(program: str) -> str:
        """Strip the idea comment from a program."""
        lines = program.splitlines(keepends=True)
        result = []
        for line in lines:
            if line.strip().startswith("// Idea prompt:"):
                continue
            result.append(line)
        return ''.join(result)

    # ------------------------------------------------------------------
    # (b) Evaluation
    # ------------------------------------------------------------------

    def evaluate_one(self, llm: Any, example: dict) -> dict:
        self._ensure_chain(llm)

        args = example.get("arguments", example)
        idea = args.get("idea", "")
        example_id = example.get("_path", example.get("metadata", {}).get("file_name", "unknown"))
        ground_truth_outcome = example.get("outcome")

        user_message = format_implement_user(idea=idea)

        try:
            generated_code = self._chain.invoke({"user_message": user_message}).strip()
        except Exception as e:
            logger.warning(f"LLM call failed for {example_id}: {e}")
            return {
                "success": False,
                "example_id": example_id,
                "idea": idea[:500],
                "verification_outcome": "ERROR",
                "error": str(e),
                "ground_truth_outcome": ground_truth_outcome,
            }

        try:
            prog = DafnyProgram(generated_code, name=example_id)
            ver = prog.verify()
            outcome = ver.outcome.name
        except Exception as e:
            logger.warning(f"Verification failed for {example_id}: {e}")
            return {
                "success": False,
                "example_id": example_id,
                "idea": idea[:500],
                "generated_code": generated_code,
                "verification_outcome": "ERROR",
                "error": str(e),
                "ground_truth_outcome": ground_truth_outcome,
            }

        return {
            "success": outcome == "SUCCESS",
            "example_id": example_id,
            "idea": idea[:500],
            "generated_code": generated_code,
            "verification_outcome": outcome,
            "verification_stdout": ver.stdout,
            "verification_stderr": ver.stderr,
            "ground_truth_outcome": ground_truth_outcome,
        }

    # ------------------------------------------------------------------
    # (c) Training records
    # ------------------------------------------------------------------

    def to_training_record(self, example: dict) -> dict | None:
        response = example.get("response")
        if not response:
            return None

        messages = reconstruct_chat_messages(
            "implement",
            example.get("arguments", {}),
        )
        return {"messages": messages, "completion": str(response)}

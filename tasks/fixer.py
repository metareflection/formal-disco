#!/usr/bin/env python3
"""
Fixer task: repair broken Dafny programs via iterative LLM-guided diff application.

Supports two data sources:
  - dfy glob: load .dfy files, filter to those that don't verify, get errors
  - pickle: load repair examples from distillation pickles

Migrated from: eval_fixer.py, eval_fixer_indist.py, fixer_distill.py
"""

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser
from dafny import DafnyProgram, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE
from prompt import system_repair, format_repair_user, reconstruct_chat_messages
from distill_common import (
    remove_hints, compute_text_diff, get_dafny_errors,
    load_verified_programs, get_content, create_agenda_pickle,
)
from tasks import EvaluationTask, load_pickle_source, load_dfy_source

logger = logging.getLogger(__name__)


class FixerTask(EvaluationTask):
    """Repair broken Dafny programs by generating and applying diffs."""

    name = "fixer"
    prompt_type = "repair"

    def __init__(
        self,
        max_attempts: int = 3,
        filter_trivial: bool = True,
        cache_path: str = ".fixer_outcome_cache.json",
        verbose: bool = False,
    ):
        self.max_attempts = max_attempts
        self.filter_trivial = filter_trivial
        self.cache_path = cache_path
        self.verbose = verbose
        self._chain = None

    def _ensure_chain(self, llm: Any) -> None:
        if self._chain is not None:
            return
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_repair(
                    example_before="{example_before}",
                    example_diff="{example_diff}",
                    example_after="{example_after}",
                ),
            ),
            (
                "human",
                format_repair_user(program="{program}", notes="{notes}"),
            ),
        ])
        self._chain = prompt | llm | CodeOutputParser()

    # ------------------------------------------------------------------
    # (a) Data extraction
    # ------------------------------------------------------------------

    def extract_examples(self, sources: list[dict]) -> list[dict]:
        """Extract fixer examples from the given sources.

        For pickle sources with raw verified programs: strips hints to create
        (broken_program, errors) -> diff examples.

        For pickle sources with distill examples: loads them directly.

        For dfy sources: loads .dfy files as programs needing repair.
        """
        examples = []

        for source in sources:
            src_type = source.get("type", "pickle")

            if src_type == "pickle":
                if source.get("prompt_types"):
                    # Load pre-made distill examples
                    examples.extend(load_pickle_source(source))
                elif source.get("extract_from_verified", False):
                    # Generate examples by stripping hints from verified programs
                    examples.extend(self._extract_from_verified(source))
                else:
                    examples.extend(load_pickle_source({
                        **source,
                        "prompt_types": ["repair"],
                    }))

            elif src_type == "dfy":
                programs = load_dfy_source(source)
                examples.extend(self._build_dfy_examples(programs))

        return examples

    def _extract_from_verified(self, source: dict) -> list[dict]:
        """Generate fixer examples by removing hints from verified programs."""
        pickle_path = Path(source["path"])
        min_hints = source.get("min_hints", 1)
        verify_stripped = source.get("verify_stripped", True)

        verified_programs = load_verified_programs(
            pickle_path, source.get("include_goal_unproven", False),
        )

        examples = []
        stats = Counter()

        for path, obj in tqdm(verified_programs, desc="Generating fixer examples"):
            program_content = get_content(obj)
            if not program_content:
                stats['skip_no_content'] += 1
                continue

            original_program, trivial_diff = True, False

            while True:
                stripped_program, num_hints = remove_hints(program_content)
                if original_program and num_hints < min_hints:
                    if original_program:
                        stats['skip_too_few_hints'] += 1
                    break

                if verify_stripped:
                    try:
                        prog = DafnyProgram(stripped_program, name="stripped")
                        ver = prog.verify()
                        trivial_diff = (ver.outcome == VerificationOutcome.SUCCESS)
                        notes = ver.stdout
                        if ver.stderr:
                            notes = f"{notes}\n\nstderr:\n{ver.stderr}"
                    except Exception:
                        stats['skip_verification_error'] += 1
                        original_trivial = True
                        continue
                else:
                    notes = "(verification not run)"

                if not trivial_diff:
                    assert stripped_program != program_content
                    diff = compute_text_diff(stripped_program, program_content)
                    assert apply_text_diff(stripped_program, diff) == program_content

                    ver_status = obj.properties.get('verification_status', 'success')

                    examples.append({
                        "prompt": "repair",
                        "arguments": {"program": stripped_program, "notes": notes},
                        "response": diff,
                        "outcome": ver_status,
                        "metadata": {
                            "source": "fixer_distill",
                            "program_path": path,
                            "hints_removed": num_hints,
                        },
                    })
                    stats['examples_created'] += 1
                program_content = stripped_program
                original_program = False

        logger.info(f"Fixer extraction stats: {dict(stats)}")
        return examples

    def _build_dfy_examples(
        self,
        programs: list[tuple[str, str]],
    ) -> list[dict]:
        """Verify programs, filter trivial ones, and build examples in one pass.

        Caches both outcome and verification output (stdout/stderr) so
        subsequent runs don't re-invoke Dafny.
        """
        cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                cache = json.load(f)

        examples = []
        for name, text in tqdm(programs, desc="Verifying programs"):
            if name in cache:
                entry = cache[name]
                # Support old cache format (just a string)
                if isinstance(entry, str):
                    outcome = entry
                    stdout = ""
                    stderr = ""
                else:
                    outcome = entry["outcome"]
                    stdout = entry.get("stdout", "")
                    stderr = entry.get("stderr", "")
            else:
                prog = DafnyProgram(text, name=name)
                ver = prog.verify()
                outcome = ver.outcome.name
                stdout = ver.stdout
                stderr = ver.stderr
                cache[name] = {"outcome": outcome, "stdout": stdout, "stderr": stderr}

                if self.cache_path:
                    with open(self.cache_path, 'w') as f:
                        json.dump(cache, f, indent=2)

            if self.filter_trivial and outcome == "SUCCESS":
                continue

            notes = f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
            examples.append({
                "prompt": "repair",
                "arguments": {"program": text, "notes": notes},
                "metadata": {"source": "dfy", "program_name": name},
            })

        return examples

    # ------------------------------------------------------------------
    # (b) Evaluation
    # ------------------------------------------------------------------

    def evaluate_one(self, llm: Any, example: dict) -> dict:
        self._ensure_chain(llm)

        args = example.get("arguments", example)
        program = args.get("program", "")
        notes = args.get("notes", "")
        name = (
            example.get("metadata", {}).get("program_name")
            or example.get("metadata", {}).get("program_path", "program")
        )

        return self._fix(
            llm, program, name,
            initial_notes=notes,
        )

    def _fix(
        self,
        llm: Any,
        prog_text: str,
        program_name: str,
        initial_notes: str = "",
    ) -> dict:
        """Iterative repair loop."""
        current_text = prog_text
        interaction_log = []
        ver_notes = initial_notes
        ver = None

        # Get initial verification if no notes provided
        if not ver_notes:
            prog = DafnyProgram(current_text, name=program_name)
            ver = prog.verify()
            if ver.outcome == VerificationOutcome.SUCCESS:
                return {
                    "success": True,
                    "program_name": program_name,
                    "num_attempts": 0,
                    "verification_outcome": "SUCCESS",
                    "final_program": current_text,
                }
            ver_notes = f"Output of dafny verify on this program:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}\n"

        for attempt in range(self.max_attempts):
            if self.verbose:
                logger.info(f"Attempt {attempt + 1}/{self.max_attempts} for {program_name}")

            llm_args = {
                "program": current_text,
                "notes": ver_notes,
                "example_diff": TEXT_DIFF_EXAMPLE,
                "example_before": TEXT_BEFORE_EXAMPLE,
                "example_after": TEXT_AFTER_EXAMPLE,
            }

            try:
                diff_text = self._chain.invoke(llm_args).strip()
                interaction = {'program': current_text, 'notes': ver_notes, 'diff': diff_text}
            except Exception as e:
                logger.warning(f"LLM call failed for {program_name}: {e}")
                interaction_log.append({**interaction, 'result': f'Error: {e}'})
                continue

            try:
                repaired_text = apply_text_diff(current_text, diff_text)
            except Exception as e:
                logger.warning(f"Failed to apply diff for {program_name}: {e}")
                interaction_log.append({**interaction, 'result': f'Error: {e}'})
                continue

            repaired_prog = DafnyProgram(repaired_text, name=program_name)
            ver = repaired_prog.verify()

            current_text = repaired_text
            ver_notes = f"Output of dafny verify on this program:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}\n"

            interaction_log.append({**interaction, 'result': repaired_text, 'result_notes': ver_notes})

            if ver.outcome == VerificationOutcome.SUCCESS:
                return {
                    "success": True,
                    "program_name": program_name,
                    "num_attempts": attempt + 1,
                    "verification_outcome": "SUCCESS",
                    "final_program": current_text,
                    "interaction_log": interaction_log,
                }

        return {
            "success": False,
            "program_name": program_name,
            "num_attempts": self.max_attempts,
            "verification_outcome": ver.outcome.name if ver else "UNKNOWN",
            "final_program": current_text,
            "interaction_log": interaction_log,
        }

    # ------------------------------------------------------------------
    # (c) Training records
    # ------------------------------------------------------------------

    def to_training_record(self, example: dict) -> dict | None:
        response = example.get("response")
        if not response:
            return None

        messages = reconstruct_chat_messages(
            "repair",
            example.get("arguments", {}),
            example_before=TEXT_BEFORE_EXAMPLE,
            example_diff=TEXT_DIFF_EXAMPLE,
            example_after=TEXT_AFTER_EXAMPLE,
        )
        return {"messages": messages, "completion": str(response)}

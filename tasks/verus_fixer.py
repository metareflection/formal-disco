#!/usr/bin/env python3
"""
Verus fixer task: repair broken Verus programs via iterative LLM-guided diff application.

Supports data sources:
  - verusbench: load tasks from Verus-Bench tasks.jsonl (unverified -> verified pairs)
  - pickle: load repair examples from distillation pickles
  - rs glob: load .rs files that fail verification

Parallel to tasks/fixer.py but uses the Verus backend instead of Dafny.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from code_output_parser import CodeOutputParser
from language import Language, Program, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE
from tasks import EvaluationTask, load_pickle_source, _to_langchain_messages

_backend = Language.VERUS.get_backend()

logger = logging.getLogger(__name__)


class VerusFixerTask(EvaluationTask):
    """Repair broken Verus programs by generating and applying diffs."""

    name = "verus_fixer"
    prompt_type = "repair"

    def __init__(
        self,
        max_attempts: int = 3,
        verbose: bool = False,
    ):
        self.max_attempts = max_attempts
        self.verbose = verbose
        self._chain = None

    def _ensure_chain(self, llm: Any) -> None:
        if self._chain is not None:
            return
        self._chain = llm | CodeOutputParser()

    # ------------------------------------------------------------------
    # (a) Data extraction
    # ------------------------------------------------------------------

    def extract_examples(self, sources: list[dict]) -> list[dict]:
        examples = []

        for source in sources:
            src_type = source.get("type", "pickle")

            if src_type == "pickle":
                examples.extend(load_pickle_source({
                    **source,
                    "prompt_types": source.get("prompt_types", ["repair"]),
                }))

            elif src_type == "verusbench":
                examples.extend(self._load_verusbench(source))

            elif src_type == "rs":
                examples.extend(self._load_rs_files(source))

        return examples

    def _load_verusbench(self, source: dict) -> list[dict]:
        """Load tasks from Verus-Bench tasks.jsonl."""
        jsonl_path = source.get("path", source.get("jsonl_path", ""))
        if not jsonl_path or not Path(jsonl_path).exists():
            logger.warning(f"Verus-Bench file not found: {jsonl_path}")
            return []

        examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)

                unverified = task.get("task")
                ground_truth = task.get("ground_truth")
                if not unverified:
                    continue

                # Get verification errors for the unverified program
                prog = Program(unverified, Language.VERUS, name=task.get("task_id", ""))
                ver = prog.verify()
                notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"

                example = {
                    "prompt": "repair",
                    "arguments": {"program": unverified, "notes": notes},
                    "metadata": {
                        "source": "verusbench",
                        "task_id": task.get("task_id", ""),
                        "bench_source": task.get("source", ""),
                        "task_path": task.get("task_path", ""),
                    },
                }
                if ground_truth:
                    example["ground_truth"] = ground_truth

                examples.append(example)

        logger.info(f"Loaded {len(examples)} tasks from Verus-Bench: {jsonl_path}")
        return examples

    def _load_rs_files(self, source: dict) -> list[dict]:
        """Load .rs files matching a glob pattern, filter to those that fail verification."""
        import glob as globmod

        pattern = source["glob"]
        files = sorted(globmod.glob(pattern, recursive=True))
        examples = []

        for filepath in tqdm(files, desc="Verifying .rs files"):
            try:
                text = Path(filepath).read_text()
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

            prog = Program(text, Language.VERUS, name=Path(filepath).stem)
            ver = prog.verify()
            if ver.outcome == VerificationOutcome.SUCCESS:
                continue

            notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"
            examples.append({
                "prompt": "repair",
                "arguments": {"program": text, "notes": notes},
                "metadata": {"source": "rs", "file_path": filepath},
            })

        logger.info(f"Loaded {len(examples)} failing .rs files from {pattern}")
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
            example.get("metadata", {}).get("task_id")
            or example.get("metadata", {}).get("file_path", "program")
        )

        return self._fix(llm, program, name, initial_notes=notes)

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
            prog = Program(current_text, Language.VERUS, name=program_name)
            ver = prog.verify()
            if ver.outcome == VerificationOutcome.SUCCESS:
                return {
                    "success": True,
                    "program_name": program_name,
                    "num_attempts": 0,
                    "verification_outcome": "SUCCESS",
                    "final_program": current_text,
                }
            ver_notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"

        for attempt in range(self.max_attempts):
            if self.verbose:
                logger.info(f"Attempt {attempt + 1}/{self.max_attempts} for {program_name}")

            msgs = _to_langchain_messages(_backend.prompt_builder.repair(
                program=current_text,
                notes=ver_notes,
                example_before=TEXT_BEFORE_EXAMPLE,
                example_diff=TEXT_DIFF_EXAMPLE,
                example_after=TEXT_AFTER_EXAMPLE,
            ))

            try:
                diff_text = self._chain.invoke(msgs).strip()
                print(f"=== DIFF OUTPUT for {program_name} ===")
                print(diff_text[:500])
                print(f"=== END DIFF ===")
            except Exception as e:
                logger.warning(f"LLM call failed for {program_name}: {e}")
                interaction_log.append({'program': current_text, 'notes': ver_notes, 'result': f'Error: {e}'})
                continue

            interaction = {'program': current_text, 'notes': ver_notes, 'diff': diff_text}

            try:
                repaired_text = apply_text_diff(current_text, diff_text)
                print(f"=== CHANGED: {repaired_text != current_text} ===")
            except Exception as e:
                logger.warning(f"Failed to apply diff for {program_name}: {e}")
                interaction_log.append({**interaction, 'result': f'Error: {e}'})
                continue

            repaired_prog = Program(repaired_text, Language.VERUS, name=program_name)
            ver = repaired_prog.verify()

            current_text = repaired_text
            ver_notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"

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

        args = example.get("arguments", {})
        messages = _backend.prompt_builder.repair(
            program=args.get("program", ""),
            notes=args.get("notes", ""),
            example_before=TEXT_BEFORE_EXAMPLE,
            example_diff=TEXT_DIFF_EXAMPLE,
            example_after=TEXT_AFTER_EXAMPLE,
        )
        return {"messages": messages, "completion": str(response)}

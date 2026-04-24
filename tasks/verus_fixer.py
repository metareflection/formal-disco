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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from code_output_parser import CodeOutputParser
from distill_common import get_content, load_verified_programs, remove_hints_verus
from language import Language, Program, VerificationOutcome
from patch import apply_text_diff, compute_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE
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
                if source.get("prompt_types"):
                    examples.extend(load_pickle_source(source))
                elif source.get("extract_from_verified", False):
                    examples.extend(self._extract_from_verified(source))
                else:
                    examples.extend(load_pickle_source({
                        **source,
                        "prompt_types": ["repair"],
                    }))

            elif src_type == "verusbench":
                examples.extend(self._load_verusbench(source))

            elif src_type == "rs":
                examples.extend(self._load_rs_files(source))

        return examples

    def extract_one(
        self,
        path: str,
        obj: Any,
        min_hints: int = 1,
        verify_stripped: bool = True,
        partial: bool = True,
    ) -> tuple[list[dict], Counter]:
        """Extract repair examples from a single verified Verus program.

        When `partial` is True (default), iteratively removes a random subset
        of hints per step, producing one (stripped_i+1 -> stripped_i) diff
        example per step until no more hints remain.

        When `partial` is False, removes every hint in a single pass and
        emits at most one example with the full diff.

        Returns (examples, stats).
        """
        examples: list[dict] = []
        stats: Counter = Counter()

        program_content = get_content(obj)
        if not program_content:
            stats['skip_no_content'] += 1
            return examples, stats

        original_program, trivial_diff = True, False

        while True:
            strip_min_hints = 1 if partial else 0
            stripped_program, num_hints = remove_hints_verus(
                program_content, min_hints=strip_min_hints,
            )

            if num_hints < min_hints:
                if original_program:
                    stats['skip_too_few_hints'] += 1
                break

            if verify_stripped:
                try:
                    prog = Program(stripped_program, Language.VERUS, name="stripped")
                    ver = prog.verify()
                    trivial_diff = (ver.outcome == VerificationOutcome.SUCCESS)
                    notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"
                except Exception:
                    stats['skip_verification_error'] += 1
                    if not partial:
                        break
                    continue
            else:
                notes = "(verification not run)"

            if not trivial_diff:
                if stripped_program == program_content:
                    stats['skip_no_change'] += 1
                    break
                diff = compute_text_diff(stripped_program, program_content)
                try:
                    roundtrip_ok = apply_text_diff(stripped_program, diff) == program_content
                except Exception:
                    roundtrip_ok = False
                if not roundtrip_ok:
                    stats['skip_diff_roundtrip_fail'] += 1
                else:
                    ver_status = obj.properties.get('verification_status', 'success')

                    examples.append({
                        "prompt": "repair",
                        "arguments": {"program": stripped_program, "notes": notes},
                        "response": diff,
                        "outcome": ver_status,
                        "language": "Verus",
                        "metadata": {
                            "source": "verus_fixer_distill",
                            "program_path": path,
                            "hints_removed": num_hints,
                        },
                    })
                    stats['examples_created'] += 1
            program_content = stripped_program
            original_program = False

            if not partial:
                break

        return examples, stats

    def _extract_from_verified(self, source: dict) -> list[dict]:
        """Generate repair examples by removing hints from verified Verus programs."""
        N_THREADS = 32

        pickle_path = Path(source["path"])
        min_hints = source.get("min_hints", 1)
        verify_stripped = source.get("verify_stripped", True)
        partial = source.get("partial", True)

        verified = load_verified_programs(
            pickle_path,
            source.get("include_goal_unproven", False),
            language="verus",
        )

        all_examples: list[dict] = []
        total_stats: Counter = Counter()

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {
                executor.submit(self.extract_one, path, obj, min_hints, verify_stripped, partial): path
                for path, obj in verified
            }
            with tqdm(total=len(futures), desc="Generating verus_fixer examples") as pbar:
                for future in as_completed(futures):
                    examples, stats = future.result()
                    all_examples.extend(examples)
                    total_stats.update(stats)
                    pbar.update(1)

        logger.info(f"Verus fixer extraction stats: {dict(total_stats)}")
        return all_examples

    def _load_verusbench(self, source: dict) -> list[dict]:
        """Load tasks from Verus-Bench tasks.jsonl.

        When `extract_training: true`, run each ground_truth through the same
        strip-hints -> diff pipeline as `_extract_from_verified` to produce
        training examples (with a `response` diff) instead of eval tasks.
        """
        jsonl_path = source.get("path", source.get("jsonl_path", ""))
        if not jsonl_path or not Path(jsonl_path).exists():
            logger.warning(f"Verus-Bench file not found: {jsonl_path}")
            return []

        extract_training = source.get("extract_training", False)

        with open(jsonl_path, "r") as f:
            tasks = [json.loads(line) for line in f if line.strip()]

        if extract_training:
            return self._extract_training_from_verusbench(tasks, source, jsonl_path)

        examples = []
        for task in tasks:
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

    def _extract_training_from_verusbench(
        self,
        tasks: list[dict],
        source: dict,
        jsonl_path: str,
    ) -> list[dict]:
        """Produce SFT examples from Verus-Bench ground truths via extract_one."""
        N_THREADS = 16

        min_hints = source.get("min_hints", 1)
        verify_stripped = source.get("verify_stripped", True)
        partial = source.get("partial", True)

        class _GTObj:
            __slots__ = ("content", "properties")

            def __init__(self, content: str):
                self.content = content.encode("utf-8")
                self.properties = {"verification_status": "success"}

        items: list[tuple[str, _GTObj]] = []
        for task in tasks:
            gt = task.get("ground_truth")
            if not gt:
                continue
            tid = task.get("task_id") or f"verusbench/{len(items)}"
            items.append((f"verusbench/{tid}", _GTObj(gt)))

        all_examples: list[dict] = []
        total_stats: Counter = Counter()

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {
                executor.submit(self.extract_one, path, obj, min_hints, verify_stripped, partial): path
                for path, obj in items
            }
            with tqdm(total=len(futures), desc="Extracting verusbench training") as pbar:
                for future in as_completed(futures):
                    examples, stats = future.result()
                    all_examples.extend(examples)
                    total_stats.update(stats)
                    pbar.update(1)

        logger.info(
            f"Verus-Bench training extraction from {jsonl_path}: "
            f"{len(items)} ground truths -> {len(all_examples)} examples. "
            f"Stats: {dict(total_stats)}"
        )
        return all_examples

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
        name = (
            example.get("metadata", {}).get("task_id")
            or example.get("metadata", {}).get("file_path", "program")
        )

        return self._fix(llm, program, name)

    def _fix(
        self,
        llm: Any,
        prog_text: str,
        program_name: str,
    ) -> dict:
        """Attempts to fix the program with k independent samples from the LLM.
        Here, k = self.max_attempts, and we effectively try pass@k."""

        current_text = prog_text
        interaction_log = []
        ver = None

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
            except Exception as e:
                logger.warning(f"LLM call failed for {program_name}: {e}")
                interaction_log.append({'program': current_text, 'notes': ver_notes, 'result': f'Error: {e}'})
                continue

            interaction = {'program': current_text, 'notes': ver_notes, 'diff': diff_text}

            try:
                repaired_text = apply_text_diff(current_text, diff_text)
            except Exception as e:
                logger.warning(f"Failed to apply diff for {program_name}: {e}")
                interaction_log.append({**interaction, 'result': f'Error: {e}'})
                continue

            repaired_prog = Program(repaired_text, Language.VERUS, name=program_name)
            ver = repaired_prog.verify()

            result_notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"
            interaction_log.append({**interaction, 'result': repaired_text, 'result_notes': result_notes})

            if ver.outcome == VerificationOutcome.SUCCESS:
                return {
                    "success": True,
                    "program_name": program_name,
                    "num_attempts": attempt + 1,
                    "verification_outcome": "SUCCESS",
                    "final_program": repaired_text,
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

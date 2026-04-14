#!/usr/bin/env python3
"""
Lemma synthesis task: fill in empty lemma bodies so the program verifies.

Iterative evaluation: hollowed program -> LLM generates body -> insert -> verify -> retry.

Migrated from: eval_lemma.py, lemma_distill.py
"""

import logging
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from code_output_parser import CodeOutputParser
from language import Language, Program, VerificationOutcome
from distill_common import get_content, load_verified_programs, create_agenda_pickle
from tasks import EvaluationTask, load_pickle_source

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert Dafny programmer. You will be given a Dafny program "
    "containing a lemma with an empty body. Your job is to fill in the lemma body "
    "so that the program verifies.\n\n"
    "Output ONLY the lemma body contents (the code that goes between the { and }). "
    "Do not include the lemma signature, the braces themselves, or any explanation.\n\n"
    "Wrap your code in a ```dafny code block."
)


def format_user_prompt(program: str, lemma_name: str, notes: str) -> str:
    return (
        f"Program:\n{program}\n\n"
        f"The lemma `{lemma_name}` has an empty body. "
        f"Fill in the body so the program verifies.\n\n"
        f"Dafny verification output on the current program:\n{notes}\n\n"
        f"Output ONLY the body contents (without the surrounding braces)."
    )


# ---------------------------------------------------------------------------
# Lemma manipulation helpers
# ---------------------------------------------------------------------------

def find_lemmas(program: str) -> list[dict]:
    """Find all lemmas in a Dafny program with their body locations."""
    lines = program.split('\n')
    lemmas = []
    i = 0

    while i < len(lines):
        match = re.match(r'^(\s*)lemma\s+(\w+)', lines[i])
        if not match:
            i += 1
            continue

        lemma_name = match.group(2)
        start_line = i

        body_start_line = None
        for j in range(i, len(lines)):
            if '{' in lines[j]:
                body_start_line = j
                break

        if body_start_line is None:
            i += 1
            continue

        brace_depth = 0
        body_end_line = None
        for j in range(body_start_line, len(lines)):
            brace_depth += lines[j].count('{') - lines[j].count('}')
            if brace_depth == 0:
                body_end_line = j
                break

        if body_end_line is None:
            i += 1
            continue

        body_lines = lines[body_start_line + 1:body_end_line]
        body_text = '\n'.join(body_lines)

        stripped_body = body_text.strip()
        body_content = re.sub(r'//.*', '', stripped_body)
        body_content = re.sub(r'/\*.*?\*/', '', body_content, flags=re.DOTALL)
        body_is_trivial = body_content.strip() == '' or body_content.strip() == '{}'

        lemmas.append({
            'name': lemma_name,
            'start_line': start_line,
            'body_start_line': body_start_line,
            'body_end_line': body_end_line,
            'body': body_text,
            'body_is_trivial': body_is_trivial,
        })

        i = body_end_line + 1

    return lemmas


def hollow_lemma(program: str, lemma: dict) -> str:
    """Replace a lemma's body with an empty body."""
    lines = program.split('\n')
    before = lines[:lemma['body_start_line'] + 1]
    after = lines[lemma['body_end_line']:]
    return '\n'.join(before + after)


def insert_lemma_body(program: str, lemma_name: str, body: str) -> str:
    """Insert a body into the hollowed lemma."""
    lines = program.split('\n')
    result = []
    i = 0
    while i < len(lines):
        if re.match(rf'\s*lemma\s+{re.escape(lemma_name)}\b', lines[i]):
            while i < len(lines):
                result.append(lines[i])
                if '{' in lines[i]:
                    break
                i += 1
            i += 1
            brace_depth = 1
            while i < len(lines):
                brace_depth += lines[i].count('{') - lines[i].count('}')
                if brace_depth <= 0:
                    if body.strip():
                        result.append(body)
                    result.append(lines[i])
                    i += 1
                    break
                i += 1
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


_code_parser = CodeOutputParser()


def extract_body_from_response(response: str) -> str:
    """Extract lemma body from LLM response, stripping markdown fences."""
    text = _code_parser.parse(response.strip())
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        text = text[1:-1]
    return text.strip()


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class LemmaSynthTask(EvaluationTask):
    """Synthesize lemma bodies for Dafny programs."""

    name = "lemma_synth"
    prompt_type = "lemma_synth"

    def __init__(
        self,
        max_attempts: int = 3,
        verbose: bool = False,
    ):
        self.max_attempts = max_attempts
        self.verbose = verbose

    # ------------------------------------------------------------------
    # (a) Data extraction
    # ------------------------------------------------------------------

    def extract_examples(self, sources: list[dict]) -> list[dict]:
        """Extract lemma synthesis examples.

        Pickle sources with prompt_types: loads pre-made examples.
        Pickle sources with extract_from_verified: generates by hollowing lemma bodies.
        """
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
                        "prompt_types": ["lemma_synth"],
                    }))

        return examples

    def extract_one(
        self,
        path: str,
        obj: Any,
        content: str,
        verify_hollowed: bool = True,
    ) -> tuple[list[dict], Counter]:
        """Extract lemma synthesis examples from a single verified program.

        Returns (examples, stats).
        """
        examples = []
        stats = Counter()

        lemmas = find_lemmas(content)
        stats['total_lemmas'] += len(lemmas)

        for lemma in lemmas:
            if lemma['body_is_trivial']:
                stats['skip_trivial_body'] += 1
                continue

            hollowed = hollow_lemma(content, lemma)

            if verify_hollowed:
                try:
                    prog = Program(hollowed, Language.DAFNY, name="hollowed")
                    ver = prog.verify()
                    if ver.outcome == VerificationOutcome.SUCCESS:
                        stats['skip_still_verifies'] += 1
                        continue
                    notes = ver.stdout
                    if ver.stderr:
                        notes = f"{notes}\n\nstderr:\n{ver.stderr}"
                except Exception:
                    stats['skip_verification_error'] += 1
                    continue
            else:
                notes = "(verification not run)"

            examples.append({
                "prompt": "lemma_synth",
                "arguments": {
                    "program": hollowed,
                    "lemma_name": lemma['name'],
                    "notes": notes,
                },
                "response": lemma['body'],
                "outcome": "success",
                "metadata": {
                    "source": "lemma_distill",
                    "program_path": path,
                    "lemma_name": lemma['name'],
                },
            })
            stats['examples_created'] += 1

        return examples, stats

    def _extract_from_verified(self, source: dict) -> list[dict]:
        """Generate lemma examples by hollowing bodies from verified programs."""
        N_THREADS = 64

        pickle_path = Path(source["path"])
        verify_hollowed = source.get("verify_hollowed", True)

        programs = load_verified_programs(
            pickle_path, source.get("include_goal_unproven", False),
        )

        verified = []
        for path, obj in programs:
            content = get_content(obj)
            if content and 'lemma ' in content:
                verified.append((path, obj, content))

        all_examples = []
        total_stats = Counter()

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {
                executor.submit(
                    self.extract_one, path, obj, content, verify_hollowed,
                ): path
                for path, obj, content in verified
            }
            with tqdm(total=len(futures), desc="Extracting lemmas") as pbar:
                for future in as_completed(futures):
                    examples, stats = future.result()
                    all_examples.extend(examples)
                    total_stats.update(stats)
                    pbar.update(1)

        logger.info(f"Lemma extraction stats: {dict(total_stats)}")
        return all_examples

    # ------------------------------------------------------------------
    # (b) Evaluation
    # ------------------------------------------------------------------

    def evaluate_one(self, llm: Any, example: dict) -> dict:
        args = example.get("arguments", example)
        program = args.get("program", "")
        lemma_name = (
            args.get("lemma_name")
            or example.get("metadata", {}).get("lemma_name", "unknown")
        )
        notes = args.get("notes", "")
        program_path = example.get("metadata", {}).get("program_path", "unknown")
        name = f"{program_path}::{lemma_name}"

        result = self._synthesize(llm, program, lemma_name, notes, name)
        result["program"] = program
        return result

    def _synthesize(
        self,
        llm: Any,
        program: str,
        lemma_name: str,
        notes: str,
        example_name: str,
    ) -> dict:
        """Iterative lemma body synthesis."""
        from langchain_core.messages import SystemMessage, HumanMessage

        current_notes = notes
        ver = None
        body = ""

        for attempt in range(self.max_attempts):
            if self.verbose:
                logger.info(f"Attempt {attempt + 1}/{self.max_attempts} for {example_name}")

            user_msg = format_user_prompt(program, lemma_name, current_notes)

            try:
                response = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ])
                body = extract_body_from_response(response.content)
            except Exception as e:
                logger.warning(f"LLM call failed for {example_name}: {e}")
                continue

            try:
                filled = insert_lemma_body(program, lemma_name, body)
            except Exception as e:
                logger.warning(f"Failed to insert body for {example_name}: {e}")
                continue

            prog = Program(filled, Language.DAFNY, name=example_name)
            ver = prog.verify()

            if ver.outcome == VerificationOutcome.SUCCESS:
                return {
                    "success": True,
                    "example_name": example_name,
                    "lemma_name": lemma_name,
                    "num_attempts": attempt + 1,
                    "verification_outcome": "SUCCESS",
                    "generated_body": body,
                }

            current_notes = f"stdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"

        return {
            "success": False,
            "example_name": example_name,
            "lemma_name": lemma_name,
            "num_attempts": self.max_attempts,
            "verification_outcome": ver.outcome.name if ver else "UNKNOWN",
            "generated_body": body,
        }

    # ------------------------------------------------------------------
    # (c) Training records
    # ------------------------------------------------------------------

    def to_training_record(self, example: dict) -> dict | None:
        response = example.get("response")
        if not response:
            return None

        args = example.get("arguments", {})
        program = args.get("program", "")
        lemma_name = args.get("lemma_name", "")
        notes = args.get("notes", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(program, lemma_name, notes)},
        ]
        return {"messages": messages, "completion": str(response)}

"""
Simple worker that generates full programs iteratively,
repairing verification failures in-place.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Literal, Optional

from agenda import Agenda, Object, Task, WorkStatus
from code_output_parser import CodeOutputParser
from language import Language, Program, VerificationOutcome

from . import Worker, _to_langchain_messages

logger = logging.getLogger(__name__)


class IterativeGenerator(Worker):
    """Generates full programs and iteratively repairs them using verifier feedback.

    Each unit of fuel:
      1. Creates a 'create-program' task for itself.
      2. Prompts an LLM to generate a program (optionally inspired by a README).
      3. Calls verifier on the program. On failure, re-prompts up to max_repair_attempts
         with the full verifier output, asking for a complete repaired program.
      4. Saves the final program and updates the task.

    Constructor:
        jsonl_path: path to a JSONL in the same format as ReadmeInspiredIdeaGenerator
                    (here optional; READMEs will not be used if not provided)
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'dafny',
        jsonl_path: Optional[str] = None,
        max_readme_chars: int = 2000,
        max_repair_attempts: int = 3,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._max_repair_attempts = int(max_repair_attempts)
        self._max_readme_chars = int(max_readme_chars)
        self._distill = distill
        self._chain = self._llm | CodeOutputParser()
        self._rng = random.Random()

        self._rows: list[dict] = []
        if jsonl_path is not None:
            self._rows = self._load_jsonl(jsonl_path)
            if not self._rows:
                raise ValueError(f"No valid rows found in JSONL: {jsonl_path}")

    async def work(self, agenda: Agenda, fuel: int) -> None:
        for _ in range(fuel):
            # Sample a README if available.
            repo: Optional[str] = None
            readme: Optional[str] = None
            if self._rows:
                row = self._rng.choice(self._rows)
                repo = row["repo"].strip()
                readme = self._truncate(row["readme"], self._max_readme_chars)

            # Create a task for this generation attempt.
            task_obj = Task(
                id="create-prog",
                type="create-program",
                properties={"repo": repo} if repo else {},
            )
            task_id = await agenda.add_task(task_obj)

            try:
                # Step 1: Generate initial program.
                msgs = _to_langchain_messages(
                    self._backend.prompt_builder.generate(repo=repo, readme=readme)
                )
                program_text = self._chain.invoke(msgs).strip()

                # Step 2: Verify and iteratively repair.
                prog = Program(program_text, Language[self._language.upper()])
                ver = self._backend.verify(prog)

                logger.info("Initial program generated (task %s), outcome: %s",
                            task_id, ver.outcome.name)

                attempt = 0
                verification_history = []

                while ver.outcome != VerificationOutcome.SUCCESS and attempt < self._max_repair_attempts:
                    attempt += 1
                    notes = self._format_verifier_output(ver.stdout, ver.stderr)
                    logger.info("Repair attempt %d/%d for task %s",
                                attempt, self._max_repair_attempts, task_id)

                    repair_msgs = _to_langchain_messages(
                        self._backend.prompt_builder.repair_full(
                            program=program_text, notes=notes,
                        )
                    )
                    program_text = self._chain.invoke(repair_msgs).strip()

                    prog = Program(program_text, Language[self._language.upper()])
                    ver = self._backend.verify(prog)
                    verification_history.append(ver.outcome.name)

                    logger.info("After repair attempt %d, outcome: %s",
                                attempt, ver.outcome.name)

                # Step 3: Save the program.
                slug = self._slugify(task_id)
                prog_path = f"programs/{slug}.{self._backend.file_extension}"
                parents = []

                prog_obj_path = await agenda.create_object(Object(
                    path=prog_path,
                    type=f"{self._language}-program",
                    parents=parents,
                    content=program_text.encode("utf-8"),
                    properties={
                        "verification_outcome": ver.outcome.name,
                        "verification_stdout": ver.stdout,
                        "verification_stderr": ver.stderr,
                        "repo": repo or "",
                        "repair_attempts": attempt,
                    },
                ))

                # Distillation.
                should_distill = (
                    self._distill == 'all' or
                    (self._distill == 'success-only' and ver.outcome == VerificationOutcome.SUCCESS)
                )
                if should_distill:
                    distill_obj = {
                        "prompt": "generate",
                        "arguments": {"repo": repo, "readme": readme},
                        "response": program_text,
                        "outcome": ver.outcome.name.lower(),
                        "repair_attempts": attempt,
                    }
                    await agenda.create_object(Object(
                        path="distil/example.json",
                        type="distill-example",
                        parents=[prog_obj_path],
                        content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                    ))

                # Save to dataset/ on success.
                if ver.outcome == VerificationOutcome.SUCCESS:
                    dataset_path = f"dataset/{os.path.basename(prog_path)}"
                    await agenda.create_object(Object(
                        path=dataset_path,
                        type=f"{self._language}-program",
                        parents=[prog_obj_path],
                        content=program_text.encode("utf-8"),
                        properties={
                            "verification_status": ver.outcome.name.lower(),
                            "parent_idea": task_id,
                            "repo": repo or "",
                        },
                    ))

                # Update the task.
                notes = {
                    "program_path": prog_obj_path,
                    "verification": verification_history,
                    "repair_attempts": attempt,
                }
                if ver.outcome == VerificationOutcome.SUCCESS:
                    await agenda.update_task(task_id, work_status=WorkStatus.DONE, new_notes=notes)
                else:
                    await agenda.update_task(task_id, work_status=WorkStatus.FAILED, new_notes=notes)

            except Exception as e:
                logger.exception("Error in IterativeGenerator for task %s", task_id)
                await agenda.update_task(task_id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

    def _format_verifier_output(self, stdout: str, stderr: str) -> str:
        parts = []
        if stdout.strip():
            parts.append(f"stdout:\n{stdout.strip()}")
        if stderr.strip():
            parts.append(f"stderr:\n{stderr.strip()}")
        return "\n\n".join(parts) or "(no output)"

    def _load_jsonl(self, path: str) -> list[dict]:
        rows: list[dict] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "repo" in obj and "readme" in obj:
                    rows.append(obj)
        logger.info("Loaded %d repositories with README from %s", len(rows), path)
        return rows

    def _truncate(self, s: str, n: int) -> str:
        if len(s) <= n:
            return s
        return s[:n - 3] + "..."

    def _slugify(self, s: str) -> str:
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("-")
        slug = "".join(out)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "task"

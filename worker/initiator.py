"""Worker that generates an idea and implements it in a single LLM call.

Combines the idea-generation and implementation steps: given a random README,
the LLM proposes an idea and produces a program in one shot. Post-processing
(verification, follow-up tasks, distillation) mirrors LLMImplementer.
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


class Initiator(Worker):
    """Samples a README, asks the LLM to ideate and implement in one shot,
    then verifies and enqueues follow-up tasks (extend or repair).

    Each unit of fuel corresponds to one initiation attempt.
    """

    def __init__(
        self,
        jsonl_path: str,
        llm: Any,
        language: str = 'dafny',
        rng: Optional[random.Random] = None,
        max_readme_chars: int = 2000,
        interest_success: float = 2.0,
        interest_goal_unproven: float = 1.5,
        interest_fail: float = 0.5,
        interest_recursion_gamma: float = 0.0,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
    ):
        self._rng = rng or random.Random()
        self._rows = self._load_jsonl(jsonl_path)
        if not self._rows:
            raise ValueError(f"No valid rows found in JSONL: {jsonl_path}")

        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._max_readme_chars = int(max_readme_chars)
        self._interest_success = float(interest_success)
        self._interest_goal_unproven = float(interest_goal_unproven)
        self._interest_fail = float(interest_fail)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill
        self._chain = self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        for _ in range(fuel):
            row = self._rng.choice(self._rows)
            repo = row.get("repo", "").strip() or "unknown/repo"
            readme = (row.get("readme") or "").strip()
            readme = self._truncate(readme, self._max_readme_chars)

            # Create a task to track this initiation attempt.
            task_obj = Task(
                id="init-prog",
                type="initiate-program",
                properties={"repo": repo},
            )
            task_id = await agenda.add_task(task_obj)
            await agenda.update_task(task_id, work_status=WorkStatus.DOING)

            try:
                # Single LLM call combining ideation and implementation.
                # This is a simplification of the IdeaGenerator and Implementer workers.
                complex_examples = await agenda.get_most_complex_programs(3)
                msgs = _to_langchain_messages(
                    self._backend.prompt_builder.initiate(repo=repo, readme=readme, complex_examples=complex_examples)
                )
                program_text = self._chain.invoke(msgs).strip()

                # Save the program object.
                slug = self._slugify(task_id)
                prog_path = f"programs/{slug}.{self._backend.file_extension}"
                prog_obj_path = await agenda.create_object(Object(
                    path=prog_path,
                    type=f"{self._language}-program",
                    parents=[],
                    content=program_text.encode("utf-8"),
                    properties={"repo": repo, "parent_idea": f"initiated/{slug}"},
                ))

                # Verify.
                prog = Program(program_text, Language[self._language.upper()], name=prog_path)

                short_prog = program_text if len(program_text) < 2000 else program_text[:2000] + "..."
                logger.info("Verifying initiated program for task %s: %s", task_id, short_prog)

                ver = self._backend.verify(prog)

                logger.info("Repo: %s", repo)
                logger.info("Generated program:\n%s", program_text)
                logger.info("Verifier stdout: %s", ver.stdout)
                logger.info("Verifier stderr: %s", ver.stderr)
                logger.info("Verification outcome for task %s: %s", task_id, ver.outcome.name)

                # Update interest.
                interest_factor = {
                    VerificationOutcome.SUCCESS: self._interest_success,
                    VerificationOutcome.GOAL_UNPROVEN: self._interest_goal_unproven,
                }.get(ver.outcome, self._interest_fail)

                # Distillation.
                should_distill = (
                    self._distill == 'all' or
                    (self._distill == 'success-only' and ver.outcome == VerificationOutcome.SUCCESS)
                )
                if should_distill:
                    distill_obj = {
                        "prompt": "initiate",
                        "arguments": {"repo": repo, "readme": readme},
                        "response": program_text,
                        "outcome": ver.outcome.name.lower(),
                    }
                    await agenda.create_object(Object(
                        path="distil/example.json",
                        type="distill-example",
                        parents=[prog_obj_path],
                        content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                    ))

                await agenda.update_object(
                    prog_obj_path,
                    interest_factor=interest_factor,
                    interest_recursion_gamma=self._interest_recursion_gamma,
                    new_properties={
                        "verification_outcome": ver.outcome.name,
                        "verification_stdout": ver.stdout,
                        "verification_stderr": ver.stderr,
                    },
                )

                status_notes = {
                    "program_path": prog_obj_path,
                    "verification": [ver.outcome.name],
                }

                if ver.outcome == VerificationOutcome.SUCCESS:
                    # Save to dataset/.
                    dataset_path = f"dataset/{os.path.basename(prog_path)}"
                    await agenda.create_object(Object(
                        path=dataset_path,
                        type=f"{self._language}-program",
                        parents=[prog_obj_path],
                        content=program_text.encode("utf-8"),
                        properties={
                            "verification_status": ver.outcome.name.lower(),
                            "repo": repo,
                            "parent_idea": f"initiated/{slug}",
                        },
                    ))

                    # Enqueue extend task.
                    await agenda.add_task(Task(
                        id="ext", type="extend",
                        properties={"program": prog_obj_path},
                        interest_dependencies=[prog_obj_path],
                    ))
                    await agenda.update_task(task_id, work_status=WorkStatus.DONE, new_notes=status_notes)
                else:
                    # Enqueue repair task.
                    await agenda.add_task(Task(
                        id="rep", type="repair",
                        properties={"program": prog_obj_path},
                        interest_dependencies=[prog_obj_path],
                    ))
                    await agenda.update_task(task_id, work_status=WorkStatus.FAILED, new_notes=status_notes)

            except Exception as e:
                logger.exception("Error in Initiator for task %s", task_id)
                await agenda.update_task(task_id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

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

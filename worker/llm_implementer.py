"""Worker that implements program ideas using an LLM."""

import json
import logging
import os
from typing import Any, Literal, Optional

from agenda import Agenda, Object, Task, WorkStatus
from code_output_parser import CodeOutputParser
from language import Language, Program, VerificationOutcome

from . import Worker, _to_langchain_messages

logger = logging.getLogger(__name__)


class LLMImplementer(Worker):
    """Consumes 'implement' tasks and uses an LLM to generate programs.

    For each task: reads the idea object, prompts the LLM via the language
    backend's prompt builder, verifies the result, and enqueues a follow-up
    'extend' (on success/goal-unproven) or 'repair' task.
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'dafny',
        attempt_priority_factor: float = 0.9,
        interest_success: float = 2.0,
        interest_goal_unproven: float = 1.5,
        interest_fail: float = 0.5,
        interest_recursion_gamma: float = 0.0,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._attempt_priority_factor = float(attempt_priority_factor)
        self._interest_success = float(interest_success)
        self._interest_goal_unproven = float(interest_goal_unproven)
        self._interest_fail = float(interest_fail)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill
        self._chain = self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="implement")
            if result is None:
                break

            task, status = result[0]

            try:
                idea_path = task.properties.get("idea")
                assert idea_path

                idea_obj = await agenda.get_object(idea_path)
                assert idea_obj and idea_obj.content

                idea_text = idea_obj.content.decode("utf-8").strip()

                msgs = _to_langchain_messages(
                    self._backend.prompt_builder.implement(idea=idea_text)
                )
                program_text = self._chain.invoke(msgs).strip()

                prog_path = f"programs/{self._slugify(task.id)}.{self._backend.file_extension}"
                prog_obj_path = await agenda.create_object(Object(
                    path=prog_path,
                    type=f"{self._language}-program",
                    parents=[idea_path],
                    content=program_text.encode("utf-8"),
                ))

                prog = Program(program_text, Language[self._language.upper()], name=prog_path)

                short_prog = program_text if len(program_text) < 2000 else program_text[:2000] + "..."
                logger.info("Verifying generated program for task %s: %s", task.id, short_prog)

                ver = self._backend.verify(prog)

                logger.info("Idea: %s", idea_text)
                logger.info("Generated program:\n%s", program_text)
                logger.info("Verifier stdout: %s", ver.stdout)
                logger.info("Verifier stderr: %s", ver.stderr)
                logger.info("Verification outcome for task %s: %s", task.id, ver.outcome.name)

                interest_factor = {
                    VerificationOutcome.SUCCESS: self._interest_success,
                    VerificationOutcome.GOAL_UNPROVEN: self._interest_goal_unproven,
                }.get(ver.outcome, self._interest_fail)

                should_distill = (
                    self._distill == 'all' or
                    (self._distill == 'success-only' and ver.outcome == VerificationOutcome.SUCCESS)
                )
                if should_distill:
                    distill_obj = {
                        "prompt": "implement",
                        "arguments": {"idea": idea_text},
                        "response": program_text,
                        "outcome": ver.outcome.name.lower(),
                    }
                    await agenda.create_object(Object(
                        path="distil/example.json",
                        type="distill-example",
                        parents=[idea_path],
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

                notes = {"program_path": prog_obj_path, "verification": ver.outcome.name}

                if ver.outcome in (VerificationOutcome.SUCCESS, VerificationOutcome.GOAL_UNPROVEN):
                    dataset_path = f"dataset/{os.path.basename(prog_path)}"
                    await agenda.create_object(Object(
                        path=dataset_path,
                        type=f"{self._language}-program",
                        parents=[prog_obj_path],
                        content=program_text.encode("utf-8"),
                        properties={
                            "verification_status": ver.outcome.name.lower(),
                            "parent_idea": idea_path,
                        },
                    ))

                    followup_type = "extend" if ver.outcome == VerificationOutcome.SUCCESS else "repair"
                    await agenda.add_task(Task(
                        id="ext", type=followup_type,
                        properties={"program": prog_obj_path},
                        interest_dependencies=[prog_obj_path],
                    ))
                    await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes=notes)
                else:
                    await agenda.add_task(Task(
                        id="rep", type="repair",
                        properties={"program": prog_obj_path},
                        interest_dependencies=[prog_obj_path],
                    ))
                    await agenda.update_task(
                        task.id,
                        work_status=WorkStatus.ATTEMPTED,
                        priority_factor=self._attempt_priority_factor,
                        new_notes=notes,
                    )

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

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

"""Worker that extends and improves programs using an LLM-generated diff."""

import json
import logging
import os
from typing import Any, Literal, Optional

from agenda import Agenda, Object, Task, WorkStatus
from code_output_parser import CodeOutputParser
from language import Language, Program, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE

from . import Worker, _to_langchain_messages
from .limits import DEFAULT_MAX_PROGRAM_TOKENS, program_within_limit

logger = logging.getLogger(__name__)


class EditorWorker(Worker):
    """Consumes 'extend' tasks and uses an LLM to expand or improve programs.

    Prompts the LLM for a diff to apply to the current program, re-verifies,
    and enqueues another 'extend' on success or a 'repair' otherwise.
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'dafny',
        interest_success_boost: float = 1.1,
        interest_recursion_gamma: float = 0.0,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
        max_program_tokens: Optional[int] = DEFAULT_MAX_PROGRAM_TOKENS,
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._interest_success_boost = float(interest_success_boost)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill
        self._max_program_tokens = max_program_tokens
        self._chain = self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="extend")
            if result is None:
                break

            task, status = result[0]

            try:
                program_path = task.properties.get("program")
                assert program_path

                prog_obj = await agenda.get_object(program_path)
                assert prog_obj and prog_obj.content

                prog_text = prog_obj.content.decode("utf-8")

                msgs = _to_langchain_messages(
                    self._backend.prompt_builder.extend(
                        program=prog_text,
                        example_before=TEXT_BEFORE_EXAMPLE,
                        example_diff=TEXT_DIFF_EXAMPLE,
                        example_after=TEXT_AFTER_EXAMPLE,
                    )
                )
                diff_text = self._chain.invoke(msgs).strip()

                try:
                    updated_text = apply_text_diff(prog_text, diff_text)
                except Exception as e:
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": str(e)})
                    fuel -= 1
                    continue

                if not updated_text.strip():
                    logger.warning("Diff produced empty program for extend task %s", task.id)
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": "resulting program is empty"})
                    fuel -= 1
                    continue

                old_prog = Program(prog_text, Language[self._language.upper()], name=program_path)
                new_prog = Program(updated_text, Language[self._language.upper()], name=program_path)
                if str(self._backend.strip(new_prog)) == str(self._backend.strip(old_prog)):
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": "diff produced no meaningful change"})
                    fuel -= 1
                    continue

                await agenda.update_object(program_path, new_content=updated_text.encode("utf-8"))

                ver = self._backend.verify(new_prog)

                logger.info("Verification outcome for extend task %s: %s", task.id, ver.outcome.name)

                await agenda.update_object(program_path, new_properties={
                    "verification_outcome": ver.outcome.name,
                    "verification_stdout": ver.stdout,
                    "verification_stderr": ver.stderr,
                })

                should_distill = (
                    self._distill == 'all' or
                    (self._distill == 'success-only' and ver.outcome == VerificationOutcome.SUCCESS)
                )
                if should_distill:
                    distill_obj = {
                        "prompt": "extend",
                        "arguments": {"program": prog_text},
                        "response": diff_text,
                        "outcome": ver.outcome.name.lower(),
                    }
                    await agenda.create_object(Object(
                        path="distil/example.json",
                        type="distill-example",
                        parents=[program_path],
                        content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                    ))

                if ver.outcome == VerificationOutcome.SUCCESS:
                    dataset_path = f"dataset/{os.path.basename(program_path)}"
                    parent_idea = prog_obj.parents[0] if prog_obj.parents else None
                    await agenda.create_object(Object(
                        path=dataset_path,
                        type=f"{self._language}-program",
                        parents=[program_path],
                        content=updated_text.encode("utf-8"),
                        properties={
                            "verification_status": ver.outcome.name.lower(),
                            "parent_idea": parent_idea,
                        },
                    ))

                status_notes = {
                    "program_path": program_path,
                    "verification": status.worker_notes.get('verification', []) + [ver.outcome.name]
                }

                if ver.outcome == VerificationOutcome.SUCCESS:
                    await agenda.update_object(
                        program_path,
                        interest_factor=self._interest_success_boost,
                        interest_recursion_gamma=self._interest_recursion_gamma,
                    )
                    if program_within_limit(updated_text, self._max_program_tokens):
                        await agenda.add_task(Task(
                            id="ext", type="extend",
                            properties={"program": program_path},
                            interest_dependencies=[program_path],
                        ))
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.DONE,
                        new_notes=status_notes,
                    )
                elif ver.outcome == VerificationOutcome.GOAL_UNPROVEN:
                    await agenda.update_object(
                        program_path,
                        interest_factor=self._interest_success_boost,
                        interest_recursion_gamma=self._interest_recursion_gamma,
                    )
                    if program_within_limit(updated_text, self._max_program_tokens):
                        await agenda.add_task(Task(
                            id="rep", type="repair",
                            properties={"program": program_path},
                            interest_dependencies=[program_path],
                        ))
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.DONE,
                        new_notes=status_notes,
                    )
                else:
                    if program_within_limit(updated_text, self._max_program_tokens):
                        await agenda.add_task(Task(
                            id="rep", type="repair",
                            properties={"program": program_path},
                            interest_dependencies=[program_path],
                        ))
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.FAILED,
                        new_notes=status_notes,
                    )

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

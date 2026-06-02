"""Worker that repairs failing programs using an LLM-generated diff."""

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


class LLMFixer(Worker):
    """Consumes 'repair' tasks and uses an LLM to fix failing programs.

    Prompts the LLM with the failing program and verifier output, applies the
    returned diff, re-verifies, and either enqueues an 'extend' task on success
    or retries repair up to max_attempts times.
    """

    def __init__(
        self,
        llm: Any,
        language: str = 'dafny',
        max_attempts: int = 3,
        attempt_priority_factor: float = 0.9,
        interest_success_boost: float = 1.2,
        interest_recursion_gamma: float = 0.0,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
        max_program_tokens: Optional[int] = DEFAULT_MAX_PROGRAM_TOKENS,
    ) -> None:
        self._llm = llm
        self._backend = Language[language.upper()].get_backend()
        self._language = language.lower()
        self._max_program_tokens = max_program_tokens
        self._max_attempts = max_attempts
        self._attempt_priority_factor = float(attempt_priority_factor)
        self._interest_success_boost = float(interest_success_boost)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill
        self._chain = self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            result = await agenda.claim_next_tasks(type="repair")
            if result is None:
                break

            task, status = result[0]

            try:
                program_path = task.properties.get("program")
                assert program_path

                prog_obj = await agenda.get_object(program_path)
                assert prog_obj and prog_obj.content

                prog_text = prog_obj.content.decode("utf-8")

                props = getattr(prog_obj, 'properties', {}) or {}
                ver_out = props.get('verification_stdout', '')
                ver_err = props.get('verification_stderr', '')
                notes = f"Verifier stdout:\n{ver_out}\n\nVerifier stderr:\n{ver_err}\n"

                msgs = _to_langchain_messages(
                    self._backend.prompt_builder.repair(
                        program=prog_text,
                        notes=notes,
                        example_before=TEXT_BEFORE_EXAMPLE,
                        example_diff=TEXT_DIFF_EXAMPLE,
                        example_after=TEXT_AFTER_EXAMPLE,
                    )
                )
                diff_text = self._chain.invoke(msgs).strip()

                try:
                    repaired_text = apply_text_diff(prog_text, diff_text)
                except Exception as e:
                    logger.warning("Failed to apply diff for repair task %s: %s", task.id, e)
                    logger.info("Diff produced by LLM:\n%s", diff_text)
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.ATTEMPTED,
                        priority_factor=self._attempt_priority_factor,
                        new_notes={"diff_error": str(e)},
                    )
                    fuel -= 1
                    continue

                if not repaired_text.strip():
                    logger.warning("Diff produced empty program for repair task %s", task.id)
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.ATTEMPTED,
                        priority_factor=self._attempt_priority_factor,
                        new_notes={"diff_error": "resulting program is empty"},
                    )
                    fuel -= 1
                    continue

                old_prog = Program(prog_text, Language[self._language.upper()], name=program_path)
                new_prog = Program(repaired_text, Language[self._language.upper()], name=program_path)
                if str(self._backend.strip(new_prog)) == str(self._backend.strip(old_prog)):
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.ATTEMPTED,
                        priority_factor=self._attempt_priority_factor,
                        new_notes={"diff_error": "diff produced no meaningful change"},
                    )
                    fuel -= 1
                    continue

                await agenda.update_object(program_path, new_content=repaired_text.encode("utf-8"))

                ver = self._backend.verify(new_prog)

                logger.info("Verification outcome for repair task %s: %s", task.id, ver.outcome.name)
                logger.info("Program before fix:\n%s", prog_text)
                logger.info("Diff produced by LLM:\n%s", diff_text)
                logger.info("Repaired program:\n%s", repaired_text)
                logger.info("Verifier stdout after fix:\n%s", ver.stdout)
                logger.info("Verifier stderr after fix:\n%s", ver.stderr)

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
                        "prompt": "repair",
                        "arguments": {"program": prog_text, "notes": notes},
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
                        content=repaired_text.encode("utf-8"),
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
                    if program_within_limit(repaired_text, self._max_program_tokens):
                        await agenda.add_task(Task(
                            id="ext", type="extend",
                            properties={"program": program_path},
                            interest_dependencies=[program_path],
                        ))
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.DONE,
                        new_notes=status_notes,
                    )
                else:
                    await agenda.update_task(
                        task.id, work_status=WorkStatus.ATTEMPTED,
                        priority_factor=self._attempt_priority_factor,
                        new_notes=status_notes,
                    )

            except Exception as e:
                logger.exception("Error processing repair task %s: %s", task.id, e)
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

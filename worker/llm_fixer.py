#!/usr/bin/env python3

from typing import Any, Optional
import logging
import json

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser

from prompt import format_repair_user, system_repair

from dafny import DafnyProgram, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE

logger = logging.getLogger(__name__)


class LLMFixer(Worker):
    """
    Worker that processes 'repair' tasks by asking an LLM to fix a Dafny program.

    Behavior:
      - Claim a 'repair' task.
      - Load the program object referenced in task.properties['program']
      - Prompt LLM with the failing program and verification outcome / notes
      - Create a new program object with repaired text and verify it.
      - On success: enqueue an 'extend' task with the new program path and mark the repair task DONE.
      - On failure: if attempts < max_attempts, enqueue another 'repair' task with incremented attempts,
        otherwise mark the repair task FAILED.
    """

    def __init__(self, llm: Any, max_attempts: int = 3, prompt_template: Optional[ChatPromptTemplate] = None, attempt_priority_factor: float = 0.9,
                 interest_success_boost: float = 1.2, interest_recursion_gamma: float = 0.0, distill: bool = False) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._attempt_priority_factor = float(attempt_priority_factor)
        self._interest_success_boost = float(interest_success_boost)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
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
                ]
            )
        else:
            self._prompt = prompt_template

        self._chain = self._prompt | self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            tasks = await agenda.get_tasks(type="repair", ignore_completed=True)
            if not tasks:
                break

            picked = None
            for t, status in tasks:
                if status.work_status in (WorkStatus.NEW, WorkStatus.ATTEMPTED):
                    picked = (t, status)
                    break

            if picked is None:
                break

            task, status = picked

            try:
                await agenda.update_status(task.id, WorkStatus.DOING)
            except RuntimeError:
                continue

            try:
                program_path = task.properties.get("program")
                assert program_path

                prog_obj = await agenda.get_object(program_path)
                assert prog_obj and prog_obj.content

                prog_text = prog_obj.content.decode("utf-8")

                # Attempt to read verification logs from object properties
                # (set by last worker to write to this program object)
                props = getattr(prog_obj, 'properties', {}) or {}
                ver_out = props.get('verification_stdout', '')
                ver_err = props.get('verification_stderr', '')

                prompt_notes = f"Output of dafny verify on this program:\nstdout:\n{ver_out}\n\nstderr:\n{ver_err}\n"

                llm_args = {
                    "program": prog_text,
                    "notes": prompt_notes,
                    "example_diff": TEXT_DIFF_EXAMPLE,
                    "example_before": TEXT_BEFORE_EXAMPLE,
                    "example_after": TEXT_AFTER_EXAMPLE,
                }

                # Ask LLM to produce a diff to repair the program.
                diff_text = self._chain.invoke(llm_args).strip()

                # Apply diff; if it fails, mark ATTEMPTED and continue
                try:
                    repaired_text = apply_text_diff(prog_text, diff_text)
                except Exception as e:

                    logger.warning("Failed to apply diff produced by LLM for repair task %s: %s", task.id, str(e))
                    logger.info("Diff produced by LLM:\n%s", diff_text)

                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, priority_factor=self._attempt_priority_factor, new_notes={"diff_error": str(e)})
                    fuel -= 1
                    continue

                # Update the existing program object in place so patch history is maintained
                await agenda.update_object(program_path, new_content=repaired_text.encode("utf-8"))

                # Verify repaired program
                repaired_prog = DafnyProgram(repaired_text, name=program_path)

                short_repaired = repaired_text if len(repaired_text) < 2000 else repaired_text[:2000] + "..."
                logger.info("Verifying repaired Dafny program for task %s: %s", task.id, short_repaired)

                ver = repaired_prog.verify()

                logger.info("Verification outcome for repair task %s: %s", task.id, ver.outcome.name)

                logger.info("Program before fix:\n%s", prog_text)
                logger.info("Dafny output before fix:\n%s", prompt_notes)
                logger.info("Diff produced by LLM:\n%s", diff_text)
                logger.info("Repaired program:\n%s", repaired_text)
                logger.info("Dafny stdout after fix:\n%s", ver.stdout)
                logger.info("Dafny stderr after fix:\n%s", ver.stderr)
                logger.info("Verification outcome after fix: %s", ver.outcome)
                # Save verification output on the program object
                await agenda.update_object(program_path, new_properties={"verification_outcome": ver.outcome.name, "verification_stdout": ver.stdout, "verification_stderr": ver.stderr})

                if self._distill:
                    distill_obj = {
                        "prompt": "repair",
                        "arguments": llm_args,
                        "response": diff_text,
                        "outcome": ver.outcome.name.lower(),
                    }
                    await agenda.create_object(
                        Object(
                            path="distil/example.json",
                            type="distill-example",
                            parents=[program_path],
                            content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                        )
                    )

                if ver.outcome == VerificationOutcome.SUCCESS:
                    # Boost interest on successful fix
                    await agenda.update_object(program_path, interest_factor=self._interest_success_boost, interest_recursion_gamma=self._interest_recursion_gamma)
                    # Enqueue extend task and mark DONE
                    follow = Task(id="ext", type="extend", properties={"program": program_path}, interest_dependencies=[program_path])
                    await agenda.add_task(follow)
                    await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes={"program_path": program_path, "verification": ver.outcome.name})
                else:
                    # Mark the existing repair task as ATTEMPTED so it can be retried later
                    # Also decrease priority slightly.
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, priority_factor=self._attempt_priority_factor, new_notes={"program_path": program_path, "verification": ver.outcome.name})

            except Exception as e:
                logger.exception("Error processing repair task %s: %s", task.id, str(e))
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

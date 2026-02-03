#!/usr/bin/env python3

from typing import Any, Literal, Optional
import logging
import json
import os

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser

from prompt import format_extend_user, system_extend

from dafny import DafnyProgram, VerificationOutcome
from patch import apply_text_diff, TEXT_DIFF_EXAMPLE, TEXT_BEFORE_EXAMPLE, TEXT_AFTER_EXAMPLE

logger = logging.getLogger(__name__)


class EditorWorker(Worker):
    """
    Worker that processes 'extend' tasks by asking an LLM to produce a text diff
    (simple diff format) to expand or improve the program.

        Behavior:
            - Claims an 'extend' task.
            - Loads the current program text from task.properties['program'].
            - Prompts the LLM to output ONLY a diff in our format (see the `patch` module), including an example.
            - Applies the diff to get updated text; updates the existing program object in-place.
            - Verifies with Dafny:
              - On SUCCESS: increases interest, marks task DONE, enqueues another 'extend'.
              - On GOAL_UNPROVEN: increases interest, marks task DONE, enqueues a 'repair'.
              - On FAILURE: saves it, marks task FAILED, enqueues a 'repair'.
            - If diff application fails (bad format), mark the task ATTEMPTED and record the error; no object changes.
    """

    def __init__(
        self,
        llm: Any,
        prompt_template: Optional[ChatPromptTemplate] = None,
        interest_success_boost: float = 1.1,
        interest_recursion_gamma: float = 0.0,
        distill: Optional[Literal['success-only', 'all']] = 'success-only',
    ) -> None:
        self._llm = llm
        self._interest_success_boost = float(interest_success_boost)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_extend(
                            example_before="{example_before}",
                            example_diff="{example_diff}",
                            example_after="{example_after}",
                        ),
                    ),
                    (
                        "human",
                        format_extend_user(program="{program}"),
                    ),
                ]
            )
        else:
            self._prompt = prompt_template

        self._chain = self._prompt | self._llm | CodeOutputParser()

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

                # Ask the LLM for a diff in our format (include example in context)
                llm_args = {
                    "program": prog_text,
                    "example_diff": TEXT_DIFF_EXAMPLE,
                    "example_before": TEXT_BEFORE_EXAMPLE,
                    "example_after": TEXT_AFTER_EXAMPLE,
                }
                diff_text = self._chain.invoke(llm_args).strip()

                try:
                    updated_text = apply_text_diff(prog_text, diff_text)
                except Exception as e:
                    # Bad diff format; count as a failed attempt on this task.
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": str(e)})
                    continue

                if not updated_text.strip():
                    # Diff resulted in empty program; count as a failed attempt.
                    logger.warning("Diff produced empty program for extend task %s", task.id)
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": "resulting program is empty"})
                    continue

                # Persist updated program in-place.
                await agenda.update_object(
                    program_path,
                    new_content=updated_text.encode("utf-8"),
                )

                # Verify updated program.
                new_prog = DafnyProgram(updated_text, name=program_path)

                short_updated = updated_text if len(updated_text) < 2000 else updated_text[:2000] + "..."
                logger.info("Verifying edited Dafny program for task %s: %s", task.id, short_updated)

                ver = new_prog.verify()

                logger.info("Verification outcome for extend task %s: %s", task.id, ver.outcome.name)

                # Save Dafny output in program properties.
                await agenda.update_object(
                    program_path,
                    new_properties={
                        "verification_outcome": ver.outcome.name,
                        "verification_stdout": ver.stdout,
                        "verification_stderr": ver.stderr,
                    },
                )

                should_distill = (
                    self._distill == 'all' or
                    (self._distill == 'success-only' and ver.outcome == VerificationOutcome.SUCCESS)
                )
                if should_distill:
                    distill_obj = {
                        "prompt": "extend",
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

                # Save to dataset folder if verification succeeded or goal unproven
                if ver.outcome == VerificationOutcome.SUCCESS or ver.outcome == VerificationOutcome.GOAL_UNPROVEN:
                    dataset_path = f"dataset/{os.path.basename(program_path)}"
                    parent_idea = prog_obj.parents[0] if prog_obj.parents else None
                    await agenda.create_object(
                        Object(
                            path=dataset_path,
                            type="dafny-program",
                            parents=[program_path],
                            content=updated_text.encode("utf-8"),
                            properties={
                                "verification_status": ver.outcome.name.lower(),
                                "parent_idea": parent_idea,
                            },
                        )
                    )

                if ver.outcome == VerificationOutcome.SUCCESS:
                    # Boost interest slightly
                    await agenda.update_object(
                        program_path,
                        interest_factor=self._interest_success_boost,
                        interest_recursion_gamma=self._interest_recursion_gamma,
                    )

                    # Enqueue a new extend task and mark DONE
                    follow = Task(
                        id="ext", type="extend", properties={"program": program_path}, interest_dependencies=[program_path]
                    )
                    await agenda.add_task(follow)
                    await agenda.update_task(
                        task.id,
                        work_status=WorkStatus.DONE,
                        new_notes={"program_path": program_path, "verification": ver.outcome.name},
                    )
                elif ver.outcome == VerificationOutcome.GOAL_UNPROVEN:
                    # Treat as success: boost interest, mark DONE, but enqueue a repair follow-up
                    await agenda.update_object(
                        program_path,
                        interest_factor=self._interest_success_boost,
                        interest_recursion_gamma=self._interest_recursion_gamma,
                    )
                    repair = Task(
                        id="rep", type="repair", properties={"program": program_path}, interest_dependencies=[program_path]
                    )
                    await agenda.add_task(repair)
                    await agenda.update_task(
                        task.id,
                        work_status=WorkStatus.DONE,
                        new_notes={"program_path": program_path, "verification": ver.outcome.name},
                    )
                else:
                    # On failure: mark this extend as FAILED, enqueue a repair task.
                    repair = Task(
                        id="rep", type="repair", properties={"program": program_path}, interest_dependencies=[program_path]
                    )
                    await agenda.add_task(repair)
                    await agenda.update_task(
                        task.id,
                        work_status=WorkStatus.FAILED,
                        new_notes={"program_path": program_path, "verification": ver.outcome.name},
                    )

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

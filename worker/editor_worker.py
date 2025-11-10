#!/usr/bin/env python3

from typing import Any, Optional
import logging

from agenda import Agenda, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser

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
    ) -> None:
        self._llm = llm
        self._interest_success_boost = float(interest_success_boost)
        self._interest_recursion_gamma = float(interest_recursion_gamma)

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert Dafny editor. You will be given a Dafny program.\n"
                            "Your job is to EXPAND or improve it by generating a diff in the following simple format:\n\n"
                            "- Lines starting with '@@' are anchors (search-forward markers). These don't modify the program, but just start a new 'block' of changes in your patch.\n"
                            "- Lines starting with '=' keep that exact line: find it forward and advance the cursor.\n"
                            "- Lines starting with '-' delete that exact line found forward.\n"
                            "- Lines starting with '+' add a new line at the current cursor.\n\n"
                            "- All lines should start with one of the special characters above and a space following them.\n"
                            "Output ONLY the diff. No explanations.\n\nHere is an example of a diff:\n\n"
                            "Text before:\n{example_before}\n\n"
                            "Example of model output (diff in the format you must follow):\n{example_diff}\n\n"
                            "Text after:\n{example_after}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Current program:\n{program}\n\n"
                            "Goal: Expand or improve the program by proposing a diff.\n"
                            "Your diff can add a new method or classes, new lemmas, tests, logical annotations (e.g., assertions, invariants, decreases, etc) to the current program, improve or edit pre/post-conditions, etc.\n"
                            "Your diff should be focused on one goal, which you are free to decide what to pursue.\n"
                            "The overall goal is to generate interesting Dafny programs for training AI assistants for a variety of tasks (synthesis, verification, edit)."
                        ),
                    ),
                ]
            )
        else:
            self._prompt = prompt_template

        self._chain = self._prompt | self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            tasks = await agenda.get_tasks(type="extend", ignore_completed=True)
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

                # Ask the LLM for a diff in our format (include example in context)
                diff_text = self._chain.invoke({
                    "program": prog_text,
                    "example_diff": TEXT_DIFF_EXAMPLE,
                    "example_before": TEXT_BEFORE_EXAMPLE,
                    "example_after": TEXT_AFTER_EXAMPLE,
                }).strip()

                try:
                    updated_text = apply_text_diff(prog_text, diff_text)
                except Exception as e:
                    # Bad diff format; count as a failed attempt on this task.
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"diff_error": str(e)})
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

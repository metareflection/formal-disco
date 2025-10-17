#!/usr/bin/env python3

from typing import Any, Optional

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dafny import DafnyProgram, VerificationOutcome


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

    def __init__(self, llm: Any, max_attempts: int = 3, prompt_template: Optional[ChatPromptTemplate] = None) -> None:
        self._llm = llm
        self._max_attempts = max_attempts

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert Dafny developer. You will be given a Dafny program that fails verification."
                            " Return a corrected, self-contained Dafny program that addresses the verification issues."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Program:\n{program}\n\n"
                            "Notes:\n{notes}\n\n"
                            "Produce only the repaired Dafny source code. Do not include commentary."
                        ),
                    ),
                ]
            )
        else:
            self._prompt = prompt_template

        self._chain = self._prompt | self._llm | StrOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        remaining = max(0, fuel)

        while remaining > 0:
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
                notes = task.notes or {}

                # Ask LLM to repair
                repaired_text = self._chain.invoke({"program": prog_text, "notes": str(notes)}).strip()

                if not repaired_text:
                    # No output - fail this attempt
                    attempts = int(task.properties.get("attempts", 0)) + 1
                    if attempts < self._max_attempts:
                        new_task = Task(id="rep", type="repair", properties={"program": program_path, "attempts": attempts})
                        await agenda.add_task(new_task)
                        await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes={"error": "empty LLM output", "attempts": attempts})
                    else:
                        await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": "empty LLM output", "attempts": attempts})

                    remaining -= 1
                    continue

                # Persist repaired program
                repaired_obj_path = await agenda.create_object(Object(path=program_path,
                                                                       type="dafny-program",
                                                                       content=repaired_text.encode("utf-8")))

                # Verify repaired program
                repaired_prog = DafnyProgram(repaired_text, name=repaired_obj_path)
                outcome = repaired_prog.verify()

                if outcome == VerificationOutcome.SUCCESS:
                    # Enqueue extend task
                    follow = Task(id="ext", type="extend", properties={"program": repaired_obj_path})
                    await agenda.add_task(follow)
                    await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes={"program_path": repaired_obj_path, "verification": outcome.name})
                else:
                    # Retry or fail based on attempts
                    attempts = int(task.properties.get("attempts", 0)) + 1
                    if attempts < self._max_attempts:
                        new_task = Task(id="rep", type="repair", properties={"program": repaired_obj_path, "attempts": attempts})
                        await agenda.add_task(new_task)
                        await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes={"program_path": repaired_obj_path, "verification": outcome.name, "attempts": attempts})
                    else:
                        await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"program_path": repaired_obj_path, "verification": outcome.name, "attempts": attempts})

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            remaining -= 1

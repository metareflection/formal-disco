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

                # Attempt to read verification logs from object properties (set by implementer)
                props = getattr(prog_obj, 'properties', {}) or {}
                ver_out = props.get('verification_stdout', '')
                ver_err = props.get('verification_stderr', '')

                prompt_notes = f"Output of dafny verify on this program:\nstdout:\n{ver_out}\n\nstderr:\n{ver_err}\n"

                # Ask LLM to repair, providing verification logs
                repaired_text = self._chain.invoke({"program": prog_text, "notes": prompt_notes}).strip()

                # Persist repaired program
                repaired_obj_path = await agenda.create_object(Object(path=program_path,
                                                                       type="dafny-program",
                                                                       content=repaired_text.encode("utf-8")))

                # Verify repaired program
                repaired_prog = DafnyProgram(repaired_text, name=repaired_obj_path)
                ver = repaired_prog.verify()

                # Save verification output on the repaired object
                await agenda.update_object(repaired_obj_path, new_properties={"verification_outcome": ver.outcome.name, "verification_stdout": ver.stdout, "verification_stderr": ver.stderr})

                if ver.outcome == VerificationOutcome.SUCCESS:
                    # Enqueue extend task and mark DONE
                    follow = Task(id="ext", type="extend", properties={"program": repaired_obj_path})
                    await agenda.add_task(follow)
                    await agenda.update_task(task.id, work_status=WorkStatus.DONE, new_notes={"program_path": repaired_obj_path, "verification": ver.outcome.name})
                else:
                    # Mark the existing repair task as ATTEMPTED so it can be retried later
                    await agenda.update_task(task.id, work_status=WorkStatus.ATTEMPTED, new_notes={"program_path": repaired_obj_path, "verification": ver.outcome.name})

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            remaining -= 1

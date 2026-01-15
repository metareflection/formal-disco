#!/usr/bin/env python3

from typing import Any, Optional
import logging
import json

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser

from prompt import system_implement, format_implement_user

from dafny import DafnyProgram, VerificationOutcome

logger = logging.getLogger(__name__)


class LLMImplementer(Worker):
    """
    Worker that takes 'implement' tasks and uses an LLM (API) to generate Dafny implementations.

    Constructor:
        llm: a LangChain Runnable/LLM (e.g., ChatOpenAI)
        prompt_template: optional ChatPromptTemplate; if not provided a sensible default is used.

    work(agenda, fuel): for each unit of fuel, claim an implement task, read the idea object,
    prompt the LLM for a Dafny implementation, create a 'dafny-program' object and mark the
    task DONE with {'program_path': ...} in notes.
    """

    def __init__(self, llm: Any, prompt_template: Optional[ChatPromptTemplate] = None, attempt_priority_factor: float = 0.9,
                 interest_success: float = 2.0, interest_goal_unproven: float = 1.5, interest_fail: float = 0.5,
                 interest_recursion_gamma: float = 0.0, distill: bool = False) -> None:
        self._llm = llm
        self._attempt_priority_factor = float(attempt_priority_factor)
        self._interest_success = float(interest_success)
        self._interest_goal_unproven = float(interest_goal_unproven)
        self._interest_fail = float(interest_fail)
        self._interest_recursion_gamma = float(interest_recursion_gamma)
        self._distill = distill

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_implement(),
                    ),
                    (
                        "human",
                        format_implement_user(idea="{idea}"),
                    ),
                ]
            )
        else:
            self._prompt = prompt_template

        self._chain = self._prompt | self._llm | CodeOutputParser()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        while fuel > 0:
            # Fetch implement tasks
            tasks = await agenda.get_tasks(type="implement", ignore_completed=True)
            if not tasks:
                break

            # pick a NEW/ATTEMPTED task
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
                # already claimed.
                continue

            try:
                idea_path = task.properties.get("idea")
                assert idea_path

                idea_obj = await agenda.get_object(idea_path)
                assert idea_obj and idea_obj.content

                idea_text = idea_obj.content.decode("utf-8").strip()

                program_text = self._chain.invoke({"idea": idea_text}).strip()
                prog_path = f"programs/{self._slugify(task.id)}.dfy"

                prog_obj_path = await agenda.create_object(Object(path=prog_path,
                                                                  type="dafny-program",
                                                                  parents=[idea_path],
                                                                  content=program_text.encode("utf-8")))

                prog = DafnyProgram(program_text, name=prog_path)

                short_prog = program_text if len(program_text) < 2000 else program_text[:2000] + "..."
                logger.info("Verifying generated Dafny program for task %s: %s", task.id, short_prog)

                ver = prog.verify()

                logger.info(f"Idea: {idea_text}")
                logger.info(f"Generated program:\n{program_text}")
                logger.info(f"Dafny stdout: {ver.stdout}")
                logger.info(f"Dafny stderr: {ver.stderr}")

                logger.info("Verification outcome for task %s: %s", task.id, ver.outcome.name)

                # Initialize interestingness based on outcome
                interest_factor = {
                    VerificationOutcome.SUCCESS: self._interest_success,
                    VerificationOutcome.GOAL_UNPROVEN: self._interest_goal_unproven,
                }.get(ver.outcome, self._interest_fail)

                # Optionally dump a distillation example after the LLM call.
                if self._distill:
                    distill_obj = {
                        "prompt": "implement",
                        "arguments": {"idea": idea_text},
                        "response": program_text,
                        "outcome": ver.outcome.name.lower(),
                    }
                    await agenda.create_object(
                        Object(
                            path="distil/example.json",
                            type="distill-example",
                            parents=[idea_path],
                            content=json.dumps(distill_obj, ensure_ascii=False).encode("utf-8"),
                        )
                    )

                # Save verification output into the program object's properties.
                await agenda.update_object(
                    prog_obj_path,
                    interest_factor=interest_factor,
                    interest_recursion_gamma=self._interest_recursion_gamma,
                    new_properties={"verification_outcome": ver.outcome.name, "verification_stdout": ver.stdout, "verification_stderr": ver.stderr})

                notes = {"program_path": prog_obj_path, "verification": ver.outcome.name}

                if ver.outcome == VerificationOutcome.SUCCESS or ver.outcome == VerificationOutcome.GOAL_UNPROVEN:
                    followup_type = "extend" if ver.outcome == VerificationOutcome.SUCCESS else "repair"
                    follow = Task(id="ext", type=followup_type, properties={"program": prog_obj_path}, interest_dependencies=[prog_obj_path])
                    await agenda.add_task(follow)
                    # Done with initial implementation of this idea.
                    await agenda.update_task(task.id,
                                             work_status=WorkStatus.DONE,
                                             new_notes=notes)
                else:
                    follow = Task(id="rep", type="repair", properties={"program": prog_obj_path}, interest_dependencies=[prog_obj_path])
                    await agenda.add_task(follow)
                    # Leave it as ATTEMPTED so it can be retried later and reduce priority
                    await agenda.update_task(task.id,
                                             work_status=WorkStatus.ATTEMPTED,
                                             priority_factor=self._attempt_priority_factor,
                                             new_notes=notes)

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            fuel -= 1

    def _slugify(self, s: str) -> str:
        # Simple slugify to produce a filesystem-friendly name
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

#!/usr/bin/env python3

from typing import Any, Optional

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dafny import DafnyProgram, VerificationOutcome


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

    def __init__(self, llm: Any, prompt_template: Optional[ChatPromptTemplate] = None) -> None:
        self._llm = llm

        if prompt_template is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert Dafny programmer. Given a short idea or specification, output a complete,"
                            " self-contained Dafny program that implements the idea. The output must be valid Dafny code and"
                            " compile/verify when possible. Keep the program concise and include any necessary helper methods."
                        ),
                    ),
                    (
                        "human",
                        (
                            "Idea/specification:\n{idea}\n\n"
                            "Produce only Dafny source code as the response. Do not include any commentary, headings, or markdown."
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
                                                                  content=program_text.encode("utf-8")))

                prog = DafnyProgram(program_text, name=prog_path)
                ver = prog.verify()

                # Save verification output into the program object's properties for later use
                await agenda.update_object(prog_obj_path, new_properties={"verification_outcome": ver.outcome.name, "verification_stdout": ver.stdout, "verification_stderr": ver.stderr})

                notes = {"program_path": prog_obj_path, "verification": ver.outcome.name}

                if ver.outcome == VerificationOutcome.SUCCESS:
                    follow = Task(id="ext", type="extend", properties={"program": prog_obj_path})
                    await agenda.add_task(follow)
                else:
                    follow = Task(id="rep", type="repair", properties={"program": prog_obj_path})
                    await agenda.add_task(follow)

                await agenda.update_task(task.id,
                                         work_status=WorkStatus.DONE,
                                         new_notes=notes)

            except Exception as e:
                await agenda.update_task(task.id, work_status=WorkStatus.FAILED, new_notes={"error": str(e)})

            remaining -= 1

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

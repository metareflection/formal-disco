import random
from typing import Optional

from agenda import Agenda, Object, Task, WorkStatus

from . import Worker

class DummyIdeaGenerator(Worker):
    """
    Example of a worker that populates the agenda with new ideas.
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    async def work(self, agenda: Agenda, fuel: int) -> None:
        for _ in range(fuel):
            r = self._rng.randint(0, 10**3)
            idea_path = f"idea/{r}.txt"
            content = f"Write a method named m{r}"

            # Create the idea object (path will be made unique by the agenda if needed).
            path = await agenda.create_object(Object(path=idea_path,
                                                     type="idea",
                                                     content=content.encode('utf-8')))

            # Enqueue a corresponding 'implement' task.
            task = Task(id="imp", type="implement", properties={'idea': path})
            await agenda.add_task(task)


class DummyImplementer(Worker):
    """
    Takes 'implement' tasks, and "performs" them:
      - Extracts the idea number from the idea path in task properties.
      - Creates 'programs/{idea-number}.dfy' with a trivial Dafny method m{idea-number}
      - Marks the task DONE (and stores the program path in notes)
    """

    async def work(self, agenda: Agenda, fuel: int) -> None:
        remaining = max(0, fuel)

        while remaining > 0:
            # Get current pending implement tasks.
            tasks = await agenda.get_tasks(type="implement", ignore_completed=True)
            if not tasks:
                break

            # Pick the first NEW/ATTEMPTED task (i.e. not completed and nobody else doing it).
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
                # Task was claimed in the meantime - try again.
                continue

            # Extract idea number.
            idea_path = task.properties.get('idea')
            assert idea_path

            idea_obj = await agenda.get_object(idea_path)
            assert idea_obj and idea_obj.content

            idea_content = idea_obj.content.decode('utf-8')
            idea_num = idea_content.split()[-1][1:]  # Idea is "Write a method named m{num}"

            # Create the program object
            prog_path = f"programs/{idea_num}.dfy"
            prog_content = (
                f"method m{idea_num}() {{\n"
                f"  print \"hello from idea {idea_num}\\n\";\n"
                f"}}\n"
            )
            prog_path = await agenda.create_object(Object(path=prog_path,
                                                          type="dafny-program",
                                                          content=prog_content.encode('utf-8')))

            await agenda.update_task(task.id,
                                     work_status=WorkStatus.DONE,
                                     new_notes={"program_path": prog_path})

            remaining -= 1

#!/usr/bin/env python3

"""
DiscoWorker agenda.

For now we only have a local implementation of the agenda, but the idea is that it will
be distributed so that we can spawn async workers on many machines.
"""

import atexit
import asyncio
import signal
import threading
import pickle
import os
import uuid
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Iterable, Protocol

from logger import AgendaLogger
from logger.noop import NoOpLogger

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class Object:
    """
    Generic representation for the inputs and outputs of tasks.

      - path: string identifier. Corresponds to a path on disk.
      - type: arbitrary object "kind" (e.g., "dafny-program")
      - parents: list of parent object IDs (for lineage)
    - content: bytes with the actual content.
    - properties: arbitrary metadata bag
    - interestingness: float multiplier used to modulate priorities of tasks that depend on this object (default 1.0)
    """
    path: str
    type: str
    parents: list[str] = field(default_factory=list)
    content: Optional[bytes] = None
    properties: dict[str, Any] = field(default_factory=dict)
    interestingness: float = 1.0


@dataclass(slots=True, kw_only=True)
class Task:
    """
    A unit of work tracked by an Agenda.

      - id: string identifier (uuid)
      - type: arbitrary task "kind" (workers can filter tasks by this)
    - parents: list of parent task IDs (for lineage / propagation)
    - properties: arbitrary metadata bag
    - interest_dependencies: list of object paths whose interestingness multiplies this task's effective priority
    """
    id: str
    type: str
    parents: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    interest_dependencies: list[str] = field(default_factory=list)


class WorkStatus(StrEnum):
    NEW = "NEW"  # Newly created, never attempted task.
    ATTEMPTED = "ATTEMPTED"  # Not completed, but some worker has already tried this.
    DOING = "DOING"  # Some worker is currently working on this.
    DONE = "DONE"  # Completed.
    FAILED = "FAILED"  # Gave up on this.


@dataclass(slots=True, kw_only=True)
class TaskStatus:
    """
    Mutable runtime status for a Task.
    """
    priority: float = 1.0
    work_status: WorkStatus = WorkStatus.NEW
    attempts: int = 0
    worker_notes: dict[str, Any] = field(default_factory=dict)

    def is_completed(self) -> bool:
        return self.work_status in (WorkStatus.DONE, WorkStatus.FAILED)


class Agenda(Protocol):
    """
    Abstract async agenda. Keeps track of tasks and a database of objects.
    """

    async def get_object(self, path: str) -> Optional[Object]:
        """
        Return the object at `path`, or None if not found.
        """
        raise NotImplementedError

    async def create_object(self, obj: Object) -> str:
        """
        Create a new object. Returns the final path.

        The path can be modified in order to be made unique (e.g., by appending a uuid).
        """
        raise NotImplementedError

    async def update_object(self,
                            path: str,
                            new_content: Optional[bytes] = None,
                            new_properties: Optional[dict[str, Any]] = None,
                            interest_factor: Optional[float] = None,
                            interest_recursion_gamma: Optional[float] = None) -> None:
        """
        Update an existing object.

        Passing None for a field keeps it as is. Properties are merged if provided.
        """
        raise NotImplementedError

    async def update_task(
        self,
        task_id: str,
        priority_factor: Optional[float] = None,
        recursion_gamma: Optional[float] = None,
        work_status: Optional[WorkStatus] = None,
        new_notes: Optional[dict[str, Any]] = None,
    ):
        """
        Updates an existing task. Passing None for a field keeps it as is.

        This should fail if updating the status to DOING and it is already 'DOING'
        (meaning the task has been claimed).
        """
        raise NotImplementedError

    async def add_task(self, task: Task) -> str:
        """
        Register a new task and return an auto-assigned id.
        """
        raise NotImplementedError

    async def get_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> list[tuple[Task, TaskStatus]]:
        """
        Return (Task, TaskStatus) pairs sorted by descending effective priority.
        If `type` is provided, filter by exact match.
        If `ignore_completed` is True, exclude DONE/FAILED.
        """
        raise NotImplementedError

    async def update_priority(
        self,
        task_id: str,
        priority_factor: float,
        recursion_gamma: float = 0.0,
    ) -> None:
        await self.update_task(
            task_id,
            priority_factor=priority_factor,
            recursion_gamma=recursion_gamma,
        )

    async def update_status(
        self,
        task_id: str,
        work_status: WorkStatus,
    ) -> None:
        await self.update_task(task_id, work_status=work_status)

    async def update_notes(
        self,
        task_id: str,
        new_notes: dict[str, Any],
    ) -> None:
        await self.update_task(task_id, new_notes=new_notes)


class LocalAgenda(Agenda):
    """
    In-memory agenda using dict/list structures and an asyncio.Lock
    for atomic multi-field updates.

    Periodically checkpoints to disk if checkpoint_path is provided.

    Notes on priority propagation (UPWARDS):
      - Setting recursion_gamma <= 0 disables propagation.
      - With recursion_gamma in (0, 1], we propagate to ancestors:
          priority(ancestor at distance d) := max(current, root_priority * gamma**d)
        where distance d >= 1 (parent: d=1, grandparent: d=2, ...).
      - We discover ancestors via the `parents` lists stored on each Task.
        If an ancestor task is not present in this agenda, that branch stops.
    """

    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            checkpoint_interval: Optional[int] = 50,
            logger: Optional[AgendaLogger] = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}
        self._status: dict[str, TaskStatus] = {}
        self._objects: dict[str, Object] = {}

        self._clock = 0
        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval = checkpoint_interval
        self._signal_ckpt_once = threading.Event()
        self._logger = logger or NoOpLogger()

        self._load()
        # FIXME: add handlers to handle SIGTERM, SIGINT, etc.
        # atexit alone is not really robust.
        atexit.register(self._checkpoint)


    def _load(self):
        if self._checkpoint_path is None:
            return
        try:
            with open(self._checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                self._tasks = data['tasks']
                self._status = data['status']
                self._objects = data.get('objects', {})
                self._clock = data.get('clock', 0)
        except FileNotFoundError:
            logger.info('No checkpoint; starting empty agenda.')
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _checkpoint(self):
        if self._checkpoint_path is None:
            return

        tmp_path = f"{self._checkpoint_path}.new"
        try:
            with open(tmp_path, 'wb') as f:
                data = {
                    'tasks': self._tasks,
                    'status': self._status,
                    'clock': self._clock,
                    'objects': self._objects,
                }
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            # Atomic rename: overwrite target with new file
            os.replace(tmp_path, self._checkpoint_path)
            logger.info(f"Checkpointed agenda to {self._checkpoint_path}.")
        except Exception as e:
            logger.warning(f"Failed to write checkpoint: {e}")
            # Clean up tmp file if something went wrong
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


    def tick(self):
        """
        Increment clock and checkpoint if needed.
        """
        self._clock += 1
        if self._checkpoint_interval and self._clock % self._checkpoint_interval == 0:
            self._checkpoint()


    def _all_ancestors(self, root_id: str) -> Iterable[tuple[str, int]]:
        """
        BFS over parents to enumerate (task_id, distance) for all ancestors.
        Distance(root) is 0; we yield only ancestors (distance >= 1).

        If a referenced parent isn't present in the agenda, we stop traversing.
        """
        queue: list[tuple[str, int]] = [(root_id, 0)]
        seen = {root_id}
        while queue:
            cur, d = queue.pop(0)
            task = self._tasks.get(cur)
            if task is None:
                continue
            for parent_id in task.parents:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                queue.append((parent_id, d + 1))
                yield (parent_id, d + 1)

    def _all_object_ancestors(self, root_path: str) -> Iterable[tuple[str, int]]:
        """
        BFS over object parents to enumerate (object_path, distance) for all ancestors.
        Distance(root) is 0; we yield only ancestors (distance >= 1).

        If a referenced parent isn't present in the agenda, we stop traversing.
        """
        queue: list[tuple[str, int]] = [(root_path, 0)]
        seen = {root_path}
        while queue:
            cur, d = queue.pop(0)
            obj = self._objects.get(cur)
            if obj is None:
                continue
            for parent_path in obj.parents:
                if parent_path in seen:
                    continue
                seen.add(parent_path)
                queue.append((parent_path, d + 1))
                yield (parent_path, d + 1)

    async def add_task(self, task: Task) -> str:
        self.tick()

        async with self._lock:
            task.id = task.id or ''

            # Ensure ID is unique.
            if not task.id or task.id in self._tasks:
                task.id += str(uuid.uuid4())

            self._tasks[task.id] = task
            self._status[task.id] = TaskStatus()

            # This task is created in the NEW state.
            self._logger.log_task_state(task.type, task.id, WorkStatus.NEW.value)

            return task.id

    async def get_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> list[tuple[Task, TaskStatus]]:
        async with self._lock:
            items: list[tuple[Task, TaskStatus]] = []
            for tid, t in self._tasks.items():
                if type is not None and t.type != type:
                    continue
                s = self._status[tid]
                if ignore_completed and s.is_completed():
                    continue
                items.append((t, s))

            # Sort by effective priority: base priority multiplied by the product
            # of interestingness of all dependency objects.
            def effective_priority(ts: tuple[Task, TaskStatus]) -> float:
                t, s = ts
                eff = s.priority
                for obj_path in getattr(t, 'interest_dependencies', []) or []:
                    obj = self._objects.get(obj_path)
                    if obj is None:
                        continue
                    try:
                        eff *= float(getattr(obj, 'interestingness', 1.0))
                    except Exception:
                        pass
                return eff

            items.sort(key=effective_priority, reverse=True)
            return items

    async def update_task(
        self,
        task_id: str,
        priority_factor: Optional[float] = None,
        recursion_gamma: Optional[float] = None,
        work_status: Optional[WorkStatus] = None,
        new_notes: Optional[dict[str, Any]] = None,
    ):
        """
        Atomic update of a task's status fields, with optional recursive
        multiplicative priority update (to this task and ancestors).

        Priority is always updated by multiplying the existing priority
        by `priority_factor`, not resetting it. The intuition is that a task
        may increase or decrease in importance depending on what workers do.

        If a worker wants to determine that a task should not be pursued anymore,
        it can set work_status to DONE or FAILED. Otherwise, the task should be
        eventually picked up based on its priority.

        Behavior:
          - If priority is not None: set the root task's priority.
          - If recursion_gamma is not None and > 0: propagate priority update
            to ancestors, multiplicatively: multiply each ancestor's priority
            by priority * gamma**distance
          - If work_status is not None: set it, and bump attempts when moving
            into ATTEMPTED, DONE, or FAILED.
          - If new_notes is provided: do a shallow merge into current worker_notes.
        """

        self.tick()

        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown task_id: {task_id}")

            status = self._status[task_id]

            if work_status == WorkStatus.DOING and status.work_status == WorkStatus.DOING:
                raise RuntimeError(f"Task {task_id} is already in status `DOING`.")

            if priority_factor is not None:
                status.priority *= priority_factor

                # Propagate priority update.
                if recursion_gamma is not None and recursion_gamma > 0:
                    for anc_id, dist in self._all_ancestors(task_id):
                        ds = self._status.get(anc_id)
                        if ds is None:
                            continue
                        propagated = priority_factor * (recursion_gamma ** dist)
                        if propagated > ds.priority:
                            ds.priority = propagated

            if work_status is not None:
                if work_status in (WorkStatus.ATTEMPTED, WorkStatus.DOING):
                    status.attempts += 1

                # Log status changes
                task = self._tasks[task_id]

                # Update status and inform logger of the new state so it can track current counts
                status.work_status = work_status
                self._logger.log_task_state(task.type, task_id, work_status.value)

            if new_notes:
                status.worker_notes.update(new_notes)

    async def get_object(self, path: str) -> Optional[Object]:
        self.tick()
        async with self._lock:
            return self._objects.get(path)

    async def create_object(self, obj: Object) -> str:
        """
        Store a new object. If path already exists, add '_{r}' before the file
        extension (or at the end if there's no extension), where r is a random
        10-hex-character suffix. Keep trying until unique.
        """
        self.tick()
        async with self._lock:
            # If no path provided, start from a UUID.
            path = obj.path or str(uuid.uuid4())

            def split_root_ext(p: str) -> tuple[str, str]:
                # Split into root + ext, where ext includes the dot (".ext"), but only
                # if the dot appears after the last '/'. Otherwise, ext = "".
                slash_i = p.rfind("/")
                dot_i = p.rfind(".")
                if dot_i > slash_i:
                    return p[:dot_i], p[dot_i:]
                return p, ""

            # If collision, append "_{r}" before extension (or at end if no ext)
            while path in self._objects:
                r = uuid.uuid4().hex[:10]
                root, ext = split_root_ext(path)
                path = f"{root}_{r}{ext}"

            obj.path = path
            self._objects[path] = obj
            return path

    async def update_object(
        self,
        path: str,
        new_content: Optional[bytes] = None,
        new_properties: Optional[dict[str, Any]] = None,
        interest_factor: Optional[float] = None,
        interest_recursion_gamma: Optional[float] = None,
    ):
        """
        Update an existing object: replace content if provided, merge properties if provided.
        If interest_factor is provided, multiply the object's interestingness by this factor.
        If interest_recursion_gamma > 0, propagate a decayed multiplicative update to ancestors
        at distance d as interest_factor * (gamma ** d).
        """
        self.tick()
        async with self._lock:
            if path not in self._objects:
                raise KeyError(f"Unknown object path: {path}")
            obj = self._objects[path]
            if new_content is not None:
                obj.content = new_content
            if new_properties:
                obj.properties.update(new_properties)
            if interest_factor is not None:
                try:
                    obj.interestingness *= float(interest_factor)
                except Exception:
                    pass
                # Propagate to ancestors if requested
                if interest_recursion_gamma is not None and interest_recursion_gamma > 0:
                    for anc_path, dist in self._all_object_ancestors(path):
                        anc = self._objects.get(anc_path)
                        if anc is None:
                            continue
                        try:
                            propagated = float(interest_factor) * (float(interest_recursion_gamma) ** dist)
                            anc.interestingness *= propagated
                        except Exception:
                            pass
